import torch
import numpy
import random
import time
from einops import rearrange


random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)

def flash_attention2():
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

    for i in range(Tr):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        mi = m_BLOCKS[i]
        li = l_BLOCKS[i]
        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]

            S_ij = Qi @ Kj.transpose(2, 3)
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
            mi_new = torch.maximum(m_block_ij, mi)
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON
            li_new = torch.exp(mi - mi_new) * li + l_block_ij
            O_i = torch.exp(mi - mi_new) * Oi + P_ij @ Vj

            print(f'-----------O{i} = attn( Q{i}, KV{j})-----------')
            print(O_i)
        O_BLOCKS[i] = O_i / li_new
        m_BLOCKS[i] = mi_new
        l_BLOCKS[i] = li_new
    O = torch.cat(O_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    print("It costs {:5f} sec".format(time.clock() - start))
    
flash_attention2()