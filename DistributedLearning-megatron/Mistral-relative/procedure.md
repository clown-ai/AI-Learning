# MoE Procedure Draft

### Define Network

```
MixtralSparseMoeBlock(
  (gate): Linear(in_features=4096, out_features=8, bias=False)
  (experts): ModuleList(
    (0-7): 8 x MixtralBlockSparseTop2MLP(
      (w1): Linear(in_features=4096, out_features=14336, bias=False)
      (w2): Linear(in_features=14336, out_features=4096, bias=False)
      (w3): Linear(in_features=4096, out_features=14336, bias=False)
      (act_fn): SiLU()
    )
  )
)

official MoE's atchitechture as following:
MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): MixtralDecoderLayer(
        (self_attn): MixtralAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): MixtralSparseMoeBlock(
          (gate): Linear(in_features=4096, out_features=8, bias=False)
          (experts): ModuleList(
            (0-7): 8 x MixtralBLockSparseTop2MLP(
              (w1): Linear(in_features=4096, out_features=14336, bias=False)
              (w2): Linear(in_features=14336, out_features=4096, bias=False)
              (w3): Linear(in_features=4096, out_features=14336, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
)
```

Each $MLP$ of MoE's architechture $FFN$ consists of 3 parts, $w_1, w_2, w_3$, and there's 8 $MLP$，so the total parameter:
$n_{vocab} * d_{model} + n_{ctx} * d_{model} + n_{layer} * (4*d_{model}^{2} + d_{ffn}*d_{model}*3*8)$
```
32000 * 4096 + 4096 * 32768 + 32 * (4096 * 4096 * 4 + 14336 * 4096 * 3 * 8) = 47509929984
```

#### Define `MixtralBlockSparseTop2MLP`: 
```
MixtralBlockSparseTop2MLP(
  (w1): Linear(in_features=128, out_features=256, bias=False)
  (w2): Linear(in_features=256, out_features=128, bias=False)
  (w3): Linear(in_features=128, out_features=256, bias=False)
  (act_fn): SiLU()
)
```

#### Define `MixtralSparseMoeBlock`:
```
MixtralSparseMoeBlock(
  (gate): Linear(in_features=128, out_features=8, bias=False)    
  (experts): ModuleList(
    (0-7): 8 x MixtralBlockSparseTop2MLP(
      (w1): Linear(in_features=128, out_features=256, bias=False)
      (w2): Linear(in_features=256, out_features=128, bias=False)
      (w3): Linear(in_features=128, out_features=256, bias=False)
      (act_fn): SiLU()
    )
  )
)
```

### Routing
Assuming that we have a sentence with 6 tokens. And now it begins enter `MLP` part. According to the model architecture, it will pass on the `gate`.
```
tokens = 6
x = torch.randn(1, tokens, 128)
hidden_states = x
batch_size, sequence_length, hidden_dim = hidden_states.shape
hidden_states = hidden_states.view(-1, hidden_dim)
```
Each layer will generate `router_logits` for `load balance loss`
```
router_logits = experts.gate(hidden_states)
```
Compute TopK expert's logits and Top2 expert's position
```
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
routing_weights, selected_expert = torch.topk(routing_weights, experts.num_experts_per_tok, dim=-1)
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
```
Compute expert mask
```
expert_mask = F.one_hot(selected_expert, num_classes=experts.num_local_experts).permute(2, 1, 0)
```

### Expert Procedure

$y$ = $\sum_{i=0}^{n-1} Softmax(Top2(x · W_g))_i · SwiGLU_i(x)$

`sMoE` based on expert to select token to compute

`token pre-order`: token selects expert to compute sMoE's result

`expert pre-order`: compute experts' result to get the token's sMoE result

Assuming that the `final_hidden_states` is (batch_size * seq_length, hidden_state). Now, every expert needs to collect the token to compute.

```
expert 0:
Ⅰ、aquire mask
expert_layer = experts.experts[expert_id]
# tensor([[0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]])

Ⅱ、select index and top_x
index, top_x = torch.where(expert_mask[expert_id])
top_x_list = top_x.tolist() # convert tensor to list
index_list = index.tolist()
# expert 0 compute sample index: []
# expert 0 top1:0, top2:1 []
# 0 / 6 token select expert 0

Ⅲ、token pre-order computing
current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
current_hidden_state = expert_layer(current_state) * routing_weights[top_x_list, index_list, None]
# current_state: torch.Size([0, 128])
# routing_weights: tensor([], size=(0, 1), grad_fn=<IndexBackward0>)
# current_hidden_state: torch.Size([0, 128])

Ⅳ、update final_result
final_hidden_states.index_add_(0, top_x, current_hidden_state.to(hidden_states.dtype))
# torch.Size([6, 128])
```

`index_add_` operation in PyTorch is a tensor in-place operation, used to add the values from another tensor to the current tensor at corresponding positions specified by indices. 
Where:
`dim` is the dimension along which the operation is performed.
`index` is a tensor containing indices, specifying where to perform the addition operation in the current tensor.
`source` is the tensor containing values to be added to the current tensor.

### Load Balance Loss

$loss$ =  &alpha; · $N · \sum_{i=1}^{N} f_i · P_i$

$f_i$ = $1/T$ · $\sum_{x∈B}${argmax $p(x)=i$}.   The probability of $i_{th}$ expert to be allocated the token in a batch

$P_i$ = $1/T$ · $\sum_{x∈B}$ $p_i(x)$.   The sum of probability that each expert selects the token in a batch
