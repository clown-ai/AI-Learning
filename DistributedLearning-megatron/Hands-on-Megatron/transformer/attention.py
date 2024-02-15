from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch

# from megatron.core import parallel_state, tensor_parallel
# from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class SelfAttentionSubModules:
    linear_qkv: Union[ModuleSpec, type] = None
    dot_product_attention: Union[ModuleSpec, type] = None
    linear_porj: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """
    Attention layer abstract class.
    
    This layer only contains common modules require for the `self-attn`
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubModules,
        layer_number: int = 1,
        attn_mask_type=AttnMaskType.padding,
        **kwargs
    ):
        super.__init__(config=config)
        
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so the num_attention_heads_per_partition == num_query_groups_per_partition --> np == ng
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per partition head and per partition values
        # hn, np, ng
        world_size = parallel_state.get_tensor_model_paralllel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.dot_product_attention = build_module(
            submodules.dot_product_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type
        )

        self.checkpoint_dot_product_attention = self.config.recompute_granularity == 'selective'

        self.linear_porj = build_module(
            submodules.linear_porj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True
        )

    def _checkpointed_attention_forward(self, query, key, value, attention_mask, rotary_pos_emb=None):
        """
        Froward method with selective activation checkpointing.
        """
        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            output_ = self.dot_product_attention(query, key, value, attention_mask)
            return output_
        
        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask, rotary_pos_emb
        )
        return hidden_states
    
    def _allocate_memory(self, inference_max_seq_length, batch_size, dtype):
        """
        Allocate memory for to store kv cache during infernce.
        """
        return torch.empty(
            inference_max_seq_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device()
        )
    
    def forward(self, hidden_states, attention_mask, key_value_states=None, inference_params=None, rotary_pos_emb=None):
        # hidden_states: [sq, b, h]
        # For self-attention, we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
            

class SelfAttention(Attention):
    """
    self-attention layer class:
    self-attention layer takes input with size [s, b, h] and returns output with the same size
    """
    def __init__(
        self, 
        config: TransformerConfig, 
        submodules: SelfAttentionSubModules, 
        layer_number: int = 1, 
        attn_mask_type=AttnMaskType.padding, 
        **kwargs
    ):
        super().__init__(
            config=config, 
            submodules=submodules, 
            layer_number=layer_number, 
            attn_mask_type=attn_mask_type, 
            **kwargs
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + self.kv_projection_size * 2,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False
        )

    def get_query_key_value_tensor(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and 'value' tensors from `hidden_states
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            )
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        
        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(
            mixed_qkv,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3
        )
        # [sq, b, ng, np/ng * hn] --> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value
