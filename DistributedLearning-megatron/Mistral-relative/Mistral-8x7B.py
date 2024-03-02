import torch
import torch.nn as nn
from transformers import MixtralConfig


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config:MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states):
        y = self.act_fn(self.w1(hidden_states)) * self.w2(hidden_states)
        y = self.w3(y)
        return y

# expert = MixtralBlockSparseTop2MLP(MixtralConfig())
# print("单个专家为原来LLAMA的MLP层")
# print(expert)
# print("\n")

# MixtralBlockSparseTop2MLP(
#   (w1): Linear(in_features=4096, out_features=14336, bias=False)
#   (w2): Linear(in_features=14336, out_features=4096, bias=False)
#   (w3): Linear(in_features=4096, out_features=14336, bias=False)
#   (act_fn): SiLU()
# )


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config:MixtralConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # 多个 SwiGLU MLP 层组成混合专家
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(MixtralConfig()) for _ in range(self.num_experts)])

# experts = MixtralSparseMoeBlock(MixtralConfig())
# print("多个专家组成混合专家")
# print(experts)

# MixtralSparseMoeBlock(
#   (gate): Linear(in_features=4096, out_features=8, bias=False)
#   (experts): ModuleList(
#     (0-7): 8 x MixtralBlockSparseTop2MLP(
#       (w1): Linear(in_features=4096, out_features=14336, bias=False)
#       (w2): Linear(in_features=14336, out_features=4096, bias=False)
#       (w3): Linear(in_features=4096, out_features=14336, bias=False)
#       (act_fn): SiLU()
#     )
#   )
# )

# 官方 MoE 的架构如下：（以 32 层为例）
# MixtralForCausalLM(
#   (model): MixtralModel(
#     (embed_tokens): Embedding(32000, 4096)
#     (layers): ModuleList(
#       (0-31): MixtralDecoderLayer(
#         (self_attn): MixtralAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): MixtralRotaryEmbedding()
#         )
#         (block_sparse_moe): MixtralSparseMoeBlock(
#           (gate): Linear(in_features=4096, out_features=8, bias=False)
#           (experts): ModuleList(
#             (0-7): 8 x MixtralBLockSparseTop2MLP(
#               (w1): Linear(in_features=4096, out_features=14336, bias=False)
#               (w2): Linear(in_features=14336, out_features=4096, bias=False)
#               (w3): Linear(in_features=4096, out_features=14336, bias=False)
#               (act_fn): SiLU()
#             )
#           )
#         )
#         (input_layernorm): MixtralRMSNorm()
#         (post_attention_layernorm): MixtralRMSNorm()
#       )
#     )
#     (norm): MixtralRMSNorm()
#   )