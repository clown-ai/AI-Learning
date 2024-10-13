from typing import Tuple
import torch
import torch.nn.functional as F


def get_cu_seqlens_and_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch, and another int32 tensor of shape [batch_size * max_seqlen, 1, 1]
    containing the indices for the valid tokens.
    """
    mask = mask.squeeze(1).squeeze(1)
    bs, seqnlen = mask.shape

    # 计算每个序列中的 padding 数量，结果是 [2, 3, 1]（分别为每个序列中 0 的数量）。
    reduced_mask = mask.logical_not().sum(dim=1)
    # 计算累计的 padding 数量，结果是 [2, 5, 6]，表示累积的 padding 位置。
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    # 前面加上一个零，使 cu_seqlens 变为 [0, 2, 5, 6]。
    zero = torch.zeros(1, dtype=torch.int32)
    cu_seqlens = torch.cat((zero, cu_seqlens))

    mask = mask.reshape(-1)
    # 获取所有有效 token 的索引，结果是 [0, 1, 4, 8, 9, 10]。
    # breakpoint()
    indices = mask.nonzero()
    indices = indices.unsqueeze(-1)
    # 获取有效 token 的数量
    num_nonzeros = indices.shape[0]
    # 计算需要填充的 0 的数量
    pad_amount = bs * seqnlen - num_nonzeros
    indices = F.pad(
        input=indices, pad=(0, 0, 0, 0, 0, pad_amount), mode="constant", value=float(bs * seqnlen)
    )

    return cu_seqlens, indices


if __name__ == "__main__":
    mask = torch.tensor([[[[1, 1, 0, 0]]],
                         [[[1, 0, 0, 0]]],
                         [[[1, 1, 1, 0]]]],
                         dtype=torch.bool)
    
    cu_seqlens, indices = get_cu_seqlens_and_indices(mask)
    print("cumulative seqlen: ", cu_seqlens)
    print("valid token index: ", indices)