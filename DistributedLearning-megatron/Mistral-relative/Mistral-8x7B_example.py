import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)


# Let's begin with single expert
class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.ffn_size = 256

        self.w1 = nn.Linear(self.hidden_size, self.ffn_size, bias=False)
        self.w2 = nn.Linear(self.ffn_size, self.hidden_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.ffn_size, bias=False)

        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states):
        y = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states) # [1, 64, 256] * [1, 64, 265] Here's dot prodoct
        y = self.w2(y)
        return y

# x = torch.randn(1, 64, 128)
# expert = MixtralBlockSparseTop2MLP()
# print(expert)
# MixtralBlockSparseTop2MLP(
#   (w1): Linear(in_features=128, out_features=256, bias=False)
#   (w2): Linear(in_features=256, out_features=128, bias=False)
#   (w3): Linear(in_features=128, out_features=256, bias=False)
#   (act_fn): SiLU()
# )
# g = expert(x)
# print("single expert input: ", x.shape)
# print("single expert output: ", g.shape)
# print("\n")
# single expert input:  torch.Size([1, 64, 128]) 
# single expert output:  torch.Size([1, 64, 128])


# Now see how the experts compose of
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.ffn_size = 256
        self.num_experts_per_tok = 2
        self.num_local_experts = 8

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_local_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP() for _ in range(self.num_local_experts)])

experts = MixtralSparseMoeBlock()
# print(experts)

# MixtralSparseMoeBlock(
#   (gate): Linear(in_features=128, out_features=8, bias=False)    
#   (experts): ModuleList(
#     (0-7): 8 x MixtralBlockSparseTop2MLP(
#       (w1): Linear(in_features=128, out_features=256, bias=False)
#       (w2): Linear(in_features=256, out_features=128, bias=False)
#       (w3): Linear(in_features=128, out_features=256, bias=False)
#       (act_fn): SiLU()
#     )
#   )
# )

# Note we don't implemente the `forward` of SMoE
# Let's disassemble it.

"""
    Gating
"""
tokens = 6
x = torch.randn(1, tokens, 128)
hidden_states = x
batch_size, sequence_length, hidden_dim = hidden_states.shape
hidden_states = hidden_states.view(-1, hidden_dim)

# each layer will generate `router_logits` for `load balance loss`
router_logits = experts.gate(hidden_states)
print(f'experts.gate outout router logits : \n {router_logits}')
print(router_logits.shape)
print("\n")
# experts.gate outout router logits : 
#  tensor([[-4.4774e-01, -8.2948e-01, -1.9950e-01, -1.2991e-01, -7.1054e-02, 4.9257e-01,  1.2125e+00, -5.4788e-01],
#         [-6.4055e-02, -7.6019e-01, -8.3802e-01,  2.3502e-01,  8.0072e-01, -1.2306e-03, -1.7597e+00, -5.2749e-01],
#         [ 1.1844e-01,  3.0758e-02,  4.2171e-01, -1.5809e-01,  1.0099e+00, -7.3591e-01,  6.6744e-02,  7.1800e-01],
#         [ 4.0744e-01, -3.3551e-01,  1.2054e+00, -1.3312e+00,  9.1366e-01, -5.5640e-01, -2.9175e-01, -5.7977e-01],
#         [ 2.2747e-01,  5.6843e-01,  6.1523e-01,  8.0434e-01, -1.4354e-02, -7.1772e-02, -2.4029e-01, -6.9526e-01],
#         [-4.5978e-01, -1.0800e+00, -1.3668e-01, -1.7422e-01,  2.8021e-01,  6.9100e-02, -4.8364e-01,  3.4854e-01]], 
#         grad_fn=<MmBackward0>)
# torch.Size([6, 8])

# compute TopK expert's logits and Top2 expert's position
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
print(f'softmax weight : \n {routing_weights}')
print(routing_weights.shape)
print("\n")
# softmax weight :
#  tensor([[0.0689, 0.0470, 0.0883, 0.0946, 0.1004, 0.1763, 0.3623, 0.0623],
#         [0.1323, 0.0659, 0.0610, 0.1784, 0.3141, 0.1408, 0.0243, 0.0832],
#         [0.1035, 0.0948, 0.1401, 0.0785, 0.2524, 0.0440, 0.0983, 0.1885],
#         [0.1474, 0.0701, 0.3275, 0.0259, 0.2446, 0.0562, 0.0733, 0.0549],
#         [0.1218, 0.1713, 0.1795, 0.2168, 0.0956, 0.0903, 0.0763, 0.0484],
#         [0.0888, 0.0478, 0.1226, 0.1181, 0.1861, 0.1507, 0.0867, 0.1992]],
#        grad_fn=<SoftmaxBackward0>)
# torch.Size([6, 8])


routing_weights, selected_expert = torch.topk(routing_weights, experts.num_experts_per_tok, dim=-1)
print(f'expert select : \n {selected_expert}')
print(f'top_k : \n {routing_weights}')
print("\n")
# top_k :
#  tensor([[0.3623, 0.1763],
#         [0.3141, 0.1784],
#         [0.2524, 0.1885],
#         [0.3275, 0.2446],
#         [0.2168, 0.1795],
#         [0.1992, 0.1861]], grad_fn=<TopkBackward0>)
# expert select :
#  tensor([[6, 5],
#         [4, 3],
#         [4, 7],
#         [2, 4],
#         [3, 2],
#         [7, 4]])

routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
print(f'topk normalization : \n {routing_weights}')
routing_weights = routing_weights.to(hidden_states.dtype)
print("\n")
# topk normalization :
#  tensor([[0.6726, 0.3274],
#         [0.6378, 0.3622],
#         [0.5725, 0.4275],
#         [0.5724, 0.4276],
#         [0.5471, 0.4529],
#         [0.5171, 0.4829]], grad_fn=<DivBackward0>)

# OneHot encode for expert

# tensor([[[0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0]],

#         [[0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0]],

#         [[0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1]],

#         [[0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0]],

#         [[0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0]],

#         [[0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1, 0, 0, 0]]])

expert_mask = F.one_hot(selected_expert, num_classes=experts.num_local_experts).permute(2, 1, 0)

# tensor([[[0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0]],

#         [[0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0]],

#         [[0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 1, 0]],

#         [[0, 0, 0, 0, 1, 0],
#          [0, 1, 0, 0, 0, 0]],

#         [[0, 1, 1, 0, 0, 0],
#          [0, 0, 0, 1, 0, 1]],

#         [[0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0]],

#         [[1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0]],

#         [[0, 0, 0, 0, 0, 1],
#          [0, 0, 1, 0, 0, 0]]])
for i in range(tokens):
    print(f'【token_{i}】\n', expert_mask[:, :, i])

# 【token_0】
#  tensor([[0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [0, 0]])
# 【token_1】
#  tensor([[0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0]])

final_hidden_states = torch.zeros(
    (batch_size * sequence_length, hidden_dim), 
    dtype=hidden_states.dtype, device=hidden_states.device
)
print(f'final moe result shape for each token: {final_hidden_states.shape}')
# final moe result shape for each token: torch.Size([6, 128])

for expert_id in range(experts.num_local_experts):
    print(f'--------expert {expert_id} --------')

    expert_layer = experts.experts[expert_id]
    print(expert_mask[expert_id])
    index, top_x = torch.where(expert_mask[expert_id])
    print(f'expert {expert_id} compute sample index:', top_x.tolist())
    print(f'expert {expert_id} top1:0, top2:1', index.tolist())
    print(f'{len(top_x)} / {x.shape[1]} token select expert {expert_id}')

    top_x_list = top_x.tolist() # convert tensor to list
    index_list = index.tolist()

    current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

    current_hidden_state = expert_layer(current_state) * routing_weights[top_x_list, index_list, None]

    final_hidden_states.index_add_(0, top_x, current_hidden_state.to(hidden_states.dtype))

    print(current_state.shape)
    print(routing_weights[top_x_list, index_list, None])
    print(current_hidden_state.shape)
    print(final_hidden_states.shape)
# --------expert 0 --------
# tensor([[0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]])
# expert 0 compute sample index: []
# expert 0 top1:0, top2:1 []
# 0 / 6 token select expert 0
# torch.Size([0, 128])
# tensor([], size=(0, 1), grad_fn=<IndexBackward0>)
# torch.Size([0, 128])
# torch.Size([6, 128])
# --------expert 1 --------
# tensor([[0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]])
# expert 1 compute sample index: []
# expert 1 top1:0, top2:1 []
# 0 / 6 token select expert 1
# torch.Size([0, 128])
# tensor([], size=(0, 1), grad_fn=<IndexBackward0>)
# torch.Size([0, 128])
# torch.Size([6, 128])
# --------expert 2 --------
# tensor([[0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 0]])
# expert 2 compute sample index: [3, 4]
# expert 2 top1:0, top2:1 [0, 1]
# 2 / 6 token select expert 2
# torch.Size([2, 128])
# tensor([[0.5724],
#         [0.4529]], grad_fn=<IndexBackward0>)
# torch.Size([2, 128])
# torch.Size([6, 128])
# --------expert 3 --------
# tensor([[0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0]])
# expert 3 compute sample index: [4, 1]
# expert 3 top1:0, top2:1 [0, 1]
# 2 / 6 token select expert 3
# torch.Size([2, 128])
# tensor([[0.5471],
#         [0.3622]], grad_fn=<IndexBackward0>)
# torch.Size([2, 128])
# torch.Size([6, 128])
# --------expert 4 --------
# tensor([[0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1]])
# expert 4 compute sample index: [1, 2, 3, 5]
# expert 4 top1:0, top2:1 [0, 0, 1, 1]
# 4 / 6 token select expert 4
# torch.Size([4, 128])
# tensor([[0.6378],
#         [0.5725],
#         [0.4276],
#         [0.4829]], grad_fn=<IndexBackward0>)
# torch.Size([4, 128])
# torch.Size([6, 128])
# --------expert 5 --------
# tensor([[0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0]])
# expert 5 compute sample index: [0]
# expert 5 top1:0, top2:1 [1]
# 1 / 6 token select expert 5
# torch.Size([1, 128])
# tensor([[0.3274]], grad_fn=<IndexBackward0>)
# torch.Size([1, 128])
# torch.Size([6, 128])
# --------expert 6 --------
# tensor([[1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]])
# expert 6 compute sample index: [0]
# expert 6 top1:0, top2:1 [0]
# 1 / 6 token select expert 6
# torch.Size([1, 128])
# tensor([[0.6726]], grad_fn=<IndexBackward0>)
# torch.Size([1, 128])
# torch.Size([6, 128])
# --------expert 7 --------
# tensor([[0, 0, 0, 0, 0, 1],
#         [0, 0, 1, 0, 0, 0]])
# expert 7 compute sample index: [5, 2]
# expert 7 top1:0, top2:1 [0, 1]
# 2 / 6 token select expert 7
# torch.Size([2, 128])
# tensor([[0.5171],
#         [0.4275]], grad_fn=<IndexBackward0>)
# torch.Size([2, 128])
# torch.Size([6, 128])