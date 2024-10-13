import torch
import torch.nn as nn
from typing import Optional

shape = [1, 16, 256]   # [b, s, h]

xq = torch.randn((1, 4, 2, 8))
xk = torch.randn((1, 4, 1, 8))
def apply_rotary_emb(xq: torch.Tensor, xk:torch.Tensor, freq_cis: torch.Tensor):
    # xq: [b, s, n, h] ==> [b, s, n, h/2]
    # xk: [b, s, n_kv, h] ==> [b, s, n_kv, h/2]
    # it transfers the contiguous two data into complex form:
    # [3.14, -2.718] ==> [3.14-2.718j]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    x = torch.rand(x)
apply_rotary_emb(xq, xk, freq_cis=0)

# It still uses GQA, so Let's simulate its synthesis
class Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = 32

        self.wq = nn.Linear(
            shape[2],
            self.n_heads * self.head_dim,
            bias=False
        )

        self.wk = nn.Linear(
            shape[2],
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wv = nn.Linear(
            shape[2],
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            shape[2],
            bias=False
        )

        # [b, s, n, h]
        self.cache_v = torch.zeros(
            (
                shape[0],
                shape[1],
                self.n_heads,
                self.head_dim
            )
        ).cuda()

        self.cache_v = torch.zeros(
            (
                shape[0],
                shape[1],
                self.n_heads,
                self.head_dim
            )
        ).cuda()
    
    def forward(self, x, start_pos, freq_cis, mask:Optional[torch.Tensor]):
        batch_size, seq_length, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_length, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_length, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_length, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freq_cis)
