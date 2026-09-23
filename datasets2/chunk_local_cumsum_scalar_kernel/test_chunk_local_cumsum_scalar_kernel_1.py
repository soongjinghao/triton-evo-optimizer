

import torch
import numpy as np

from chunk_local_cumsum_scalar_kernel import chunk_local_cumsum_scalar

def pytorch_local_cumsum(x: torch.Tensor, chunk_size: int, reverse: bool = False):
    """
    Pure-PyTorch version that does exactly what the Triton kernel claims:
    cumsum inside non-overlapping chunks of size `chunk_size`.
    Shape is (B, T, H)  (batch-first, the default used by the wrapper).
    """
    B, T, H = x.shape
    assert T % chunk_size == 0
    x = x.view(B, T // chunk_size, chunk_size, H)
    if reverse:
        x = torch.flip(x, dims=[2])
    out = torch.cumsum(x, dim=2)
    if reverse:
        out = torch.flip(out, dims=[2])
    return out.view(B, T, H)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "npu"
    B, T, H = 4, 128, 32
    chunk_size = 32
    reverse = False

    g = torch.randn(B, T, H, device=device, dtype=torch.float32)

    tri_out = chunk_local_cumsum_scalar(
        g, chunk_size=chunk_size, reverse=reverse,
        cu_seqlens=None, head_first=False, output_dtype=torch.float32
    )

    ref_out = pytorch_local_cumsum(g, chunk_size=chunk_size, reverse=reverse)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-5)
    print("Single random case PASSED")
