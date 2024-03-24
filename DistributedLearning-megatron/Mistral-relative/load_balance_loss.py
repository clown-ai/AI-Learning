import torch
import random
import numpy
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)


batch_size = 10
seq_length = 6
num_experts = 8
top_k = 2

print(f"sMoE num_experts:{num_experts} top_k:{top_k} batch_size:{batch_size} seq_length:{seq_length}")

router_logits_1 = torch.randn(batch_size, seq_length, num_experts).view(-1, num_experts)
router_logits_2 = torch.randn(batch_size, seq_length, num_experts).view(-1, num_experts)
router_logits = [router_logits_1, router_logits_2]

concatenated_gate_logits = torch.cat(router_logits, dim=0)
print("单层gating的路由logits:", router_logits_1.shape)
print("双层gating的路由logits:", concatenated_gate_logits.shape)

print("根据 logits top-k 计算独热编码")
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
# print("routing_weights: ", routing_weights)
routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
# print("选中的专家", selected_experts)
expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
print(expert_mask.shape)

tokens_sum_expert = torch.sum(expert_mask.float(), dim=0)
tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
print(f"top1 每个专家平均处理的token  :", tokens_sum_expert[0])
print(f"top2 每个专家平均处理的token fi:", tokens_per_expert[1])
print(f"top1 与 top2 水平合计", tokens_per_expert.sum(dim=1))

router_prob_per_expert = torch.mean(routing_weights, dim=0)
print('router_prob_per_expert pi:', router_prob_per_expert)

print("每个专家的负载: ", tokens_per_expert * router_prob_per_expert.unsqueeze(0))
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
print(f"final loss: {overall_loss}")

# sMoE num_experts:8 top_k:2 batch_size:10 seq_length:6
# 单层gating的路由logits: torch.Size([60, 8])
# 双层gating的路由logits: torch.Size([120, 8])
# 根据 logits top-k 计算独热编码
# torch.Size([120, 2, 8])
# top1 每个专家平均处理的token  : tensor([20., 11., 11., 15., 13., 18., 14., 18.])
# top2 每个专家平均处理的token fi: tensor([0.1250, 0.1583, 0.1333, 0.1583, 0.1333, 0.0917, 0.0917, 0.1083])
# top1 与 top2 水平合计 tensor([1.0000, 1.0000])
# router_prob_per_expert pi: tensor([0.1339, 0.1148, 0.1174, 0.1361, 0.1144, 0.1329, 0.1096, 0.1409])
# 每个专家的负载:  tensor([[0.0223, 0.0105, 0.0108, 0.0170, 0.0124, 0.0199, 0.0128, 0.0211],
#         [0.0167, 0.0182, 0.0157, 0.0216, 0.0152, 0.0122, 0.0100, 0.0153]])
# final loss: 0.25172823667526245
