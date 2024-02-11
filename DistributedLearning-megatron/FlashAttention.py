import torch
import numpy
import random
import time
from einops import rearrange


random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)

def softmax():
    A = torch.randn(2, 6)
    print(A)

    A_exp = torch.exp(A)
    print(A_exp)

    A_sum = torch.sum(A_exp, dim=-1).unsqueeze(-1)
    print(A_sum)
    print(A_sum.size())

    P = A_exp / A_sum
    print(P)
# softmax()

def online_softmax():
    # 引入两个统计量 m, l
    N = 6
    m = torch.tensor(float("-inf"))
    l = 0
    x = torch.randn(N)
    a = torch.zeros(N)

    for i in range(N):
        m_pre = m
        m = torch.max(m_pre, x[i])
        l = l * (m_pre - m).exp() + (x[i] - m).exp()
    
    for i in range(N):
        a[i] = (x[i] - m).exp() / l
    
    print("x:", x)
    print("online softmax a:", a)
    print(torch.sum(a))
# online_softmax()

def flash_attention():
    start = time.clock()
    NEG_INF = float("-inf")
    EPSILON = 1e-10
    
    Q_LEN = 6
    K_LEN = 6
    Q_BLOCK_SIZE = 3
    KV_BLOCK_SIZE = 3
    Tr = Q_LEN // Q_BLOCK_SIZE
    Tc = K_LEN // KV_BLOCK_SIZE
    
    Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
    K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    O = torch.zeros_like(Q, requires_grad=True)
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF
    l = torch.zeros(Q.shape[:-1])[..., None]

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))

    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            mi = m_BLOCKS[i]
            li = l_BLOCKS[i]

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdim=True) # return values and indices
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)
            
            mi_new = torch.maximum(m_block_ij, mi)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

            O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            print(f'-----------Attn : Q{i} x K{j}-----------')
            print(O_BLOCKS[0])
            print(O_BLOCKS[1])
            print("\n")

            m_BLOCKS[i] = mi_new
            l_BLOCKS[i] = li_new
    
    O = torch.cat(O_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)

    print('It costs {:5f} sec'.format(time.clock() - start))

    # print(S_ij)
    # print(S_ij.shape)
    # print(m_block_ij)
    # print(m_block_ij[0].shape)

flash_attention()