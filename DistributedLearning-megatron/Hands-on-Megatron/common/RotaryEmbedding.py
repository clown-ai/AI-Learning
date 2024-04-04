import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, kv_channel: int, rotary_percent: float, seq_len_interpolation_factor: float=None, rotary_base=10000):
        super().__init__()

        dim = kv_channel
        if rotary_percent < 1.0:
            dim = int(kv_channel * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        self.inv_freq = 1.0 / (
            self.rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) 
                / dim
            )
        )

    def forward(self, max_seq_len, offset=0):
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor
        
        freqs = torch.outer(seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim], expand the shape
        emb = emb[:, None, None, :]

        return emb


def _rotary_half(x: Tensor):
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor):
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    t = t.to(freqs.device)
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype).to(t.device)
    sin_ = torch.sin(freqs).to(t.dtype).to()

    t = (t * cos_) + (_rotary_half(t) * sin_)

    return torch.cat((t, t_pass), dim=-1)



# example
import random
random.seed(42)
torch.manual_seed(42)


rotary_embdding = RotaryEmbedding(kv_channel=8, rotary_percent=1.0)

max_seq_length = 6
emb = rotary_embdding(max_seq_length)

# self.inv_freq
# tensor([1.0000, 0.1000, 0.0100, 0.0010], device='cuda:0')

# seq
# tensor([0., 1., 2., 3., 4., 5.], device='cuda:0')

# freqs
# tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.0000e+00, 1.0000e-01, 1.0000e-02, 1.0000e-03],
#         [2.0000e+00, 2.0000e-01, 2.0000e-02, 2.0000e-03],
#         [3.0000e+00, 3.0000e-01, 3.0000e-02, 3.0000e-03],
#         [4.0000e+00, 4.0000e-01, 4.0000e-02, 4.0000e-03],
#         [5.0000e+00, 5.0000e-01, 5.0000e-02, 5.0000e-03]], device='cuda:0')
# torch.Size([6, 4])

# emb --> this is concatenate by two `freqs` 
# tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [1.0000e+00, 1.0000e-01, 1.0000e-02, 1.0000e-03, 1.0000e+00, 1.0000e-01,
#          1.0000e-02, 1.0000e-03],
#         [2.0000e+00, 2.0000e-01, 2.0000e-02, 2.0000e-03, 2.0000e+00, 2.0000e-01,
#          2.0000e-02, 2.0000e-03],
#         [3.0000e+00, 3.0000e-01, 3.0000e-02, 3.0000e-03, 3.0000e+00, 3.0000e-01,
#          3.0000e-02, 3.0000e-03],
#         [4.0000e+00, 4.0000e-01, 4.0000e-02, 4.0000e-03, 4.0000e+00, 4.0000e-01,
#          4.0000e-02, 4.0000e-03],
#         [5.0000e+00, 5.0000e-01, 5.0000e-02, 5.0000e-03, 5.0000e+00, 5.0000e-01,
#          5.0000e-02, 5.0000e-03]], device='cuda:0')
# torch.Size([6, 8])

# emb --> this is as we have introduced, it is expand shape to 4-D
# tensor([[[[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00, 0.0000e+00, 0.0000e+00]]],


#         [[[1.0000e+00, 1.0000e-01, 1.0000e-02, 1.0000e-03, 1.0000e+00,
#            1.0000e-01, 1.0000e-02, 1.0000e-03]]],


#         [[[2.0000e+00, 2.0000e-01, 2.0000e-02, 2.0000e-03, 2.0000e+00,
#            2.0000e-01, 2.0000e-02, 2.0000e-03]]],


#         [[[3.0000e+00, 3.0000e-01, 3.0000e-02, 3.0000e-03, 3.0000e+00,
#            3.0000e-01, 3.0000e-02, 3.0000e-03]]],


#         [[[4.0000e+00, 4.0000e-01, 4.0000e-02, 4.0000e-03, 4.0000e+00,
#            4.0000e-01, 4.0000e-02, 4.0000e-03]]],


#         [[[5.0000e+00, 5.0000e-01, 5.0000e-02, 5.0000e-03, 5.0000e+00,
#            5.0000e-01, 5.0000e-02, 5.0000e-03]]]], device='cuda:0')
# torch.Size([6, 1, 1, 8])


# here 4 is our model's seq_length
t = torch.randn(8)

res = apply_rotary_pos_emb(t, emb)

# t
# tensor([ 0.3367,  0.1288,  0.2345,  0.2303, -1.1229, -0.1863,  2.2082, -0.6380],
#        device='cuda:0')

# freqs.shape[-1]
# 8
# rot_dim
# 8

# t
# tensor([ 0.3367,  0.1288,  0.2345,  0.2303, -1.1229, -0.1863,  2.2082, -0.6380],
#        device='cuda:0')
# t_pass
# tensor([], device='cuda:0)     OK, it seems that we have arrived the ideal state

# cos_
# tensor([[[[ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
#             1.0000]]],


#         [[[ 0.5403,  0.9950,  0.9999,  1.0000,  0.5403,  0.9950,  0.9999,
#             1.0000]]],


#         [[[-0.4161,  0.9801,  0.9998,  1.0000, -0.4161,  0.9801,  0.9998,
#             1.0000]]],


#         [[[-0.9900,  0.9553,  0.9996,  1.0000, -0.9900,  0.9553,  0.9996,
#             1.0000]]],


#         [[[-0.6536,  0.9211,  0.9992,  1.0000, -0.6536,  0.9211,  0.9992,
#             1.0000]]],


#         [[[ 0.2837,  0.8776,  0.9988,  1.0000,  0.2837,  0.8776,  0.9988,
#             1.0000]]]], device='cuda:0')

# sin_
# tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#             0.0000]]],


#         [[[ 0.8415,  0.0998,  0.0100,  0.0010,  0.8415,  0.0998,  0.0100,
#             0.0010]]],


#         [[[ 0.9093,  0.1987,  0.0200,  0.0020,  0.9093,  0.1987,  0.0200,
#             0.0020]]],


#         [[[ 0.1411,  0.2955,  0.0300,  0.0030,  0.1411,  0.2955,  0.0300,
#             0.0030]]],


#         [[[-0.7568,  0.3894,  0.0400,  0.0040, -0.7568,  0.3894,  0.0400,
#             0.0040]]],


#         [[[-0.9589,  0.4794,  0.0500,  0.0050, -0.9589,  0.4794,  0.0500,
#             0.0050]]]], device='cuda:0')

# _rotary_half(t)
# tensor([ 1.1229,  0.1863, -2.2082,  0.6380,  0.3367,  0.1288,  0.2345,  0.2303])

# t
# tensor([[[[ 0.3367,  0.1288,  0.2345,  0.2303, -1.1229, -0.1863,  2.2082,
#            -0.6380]]],


#         [[[ 1.1268,  0.1468,  0.2124,  0.2310, -0.3234, -0.1725,  2.2104,
#            -0.6378]]],


#         [[[ 0.8809,  0.1633,  0.1903,  0.2316,  0.7734, -0.1570,  2.2124,
#            -0.6375]]],


#         [[[-0.1749,  0.1781,  0.1681,  0.2322,  1.1591, -0.1399,  2.2142,
#            -0.6373]]],


#         [[[-1.0699,  0.1912,  0.1460,  0.2329,  0.4791, -0.1215,  2.2158,
#            -0.6371]]],


#         [[[-0.9812,  0.2024,  0.1238,  0.2335, -0.6414, -0.1018,  2.2172,
#            -0.6368]]]], device='cuda:0')