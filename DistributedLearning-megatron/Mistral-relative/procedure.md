# MoE Procedure Draft

### Define Network

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

\[y = \sum_{i=0}^{n-1} Softmax(Top2(x · W_g))_i · SwiGLU_i(x) \]

`sMoE` based on expert to select token to compute
`token pre-order`: token selects expert to compute sMoE's result
`expert pre-order`: compute experts' result to get the token's sMoE result

### Load Balance Loss
$loss$ =  &alpha; · $N · \sum_{i=1}^{N} f_i · P_i$
$f_i$ = $1/T$ · $\sum_{x∈B}${argmax $p(x)=i$}.   The probability of $i_{th}$ expert to be allocated the token in a batch
$P_i$ = $1/T$ · $\sum_{x∈B}$ $p_i(x)$.   The sum of probability that each expert selects the token in a batch

