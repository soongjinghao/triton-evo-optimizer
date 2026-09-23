# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import numpy as np

# Import the translated function
from chunk_local_cumsum_vector_kernel import chunk_local_cumsum_vector

# ---------- reference PyTorch version ------------------------------
def pytorch_local_cumsum_vector(x: torch.Tensor, chunk_size: int, reverse: bool = False):
    """
    Chunk-local cumsum along the time dimension for a 4-D tensor.
    Shape:  (B, T, H, S)  (batch-first, the wrapper default)
    """
    B, T, H, S = x.shape
    assert T % chunk_size == 0                     # keep the test simple
    x = x.view(B, T // chunk_size, chunk_size, H, S)
    if reverse:
        x = torch.flip(x, dims=[2])
    out = torch.cumsum(x, dim=2)
    if reverse:
        out = torch.flip(out, dims=[2])
    return out.view(B, T, H, S)


# -------------- quick smoke test -----------------------------------
if __name__ == "__main__":
    torch.manual_seed(1)
    device = "npu"
    B, T, H, S = 4, 128, 16, 64
    chunk_size = 32
    reverse = False

    # create input
    g = torch.randn(B, T, H, S, device=device, dtype=torch.float32)

    # Triton result
    tri_out = chunk_local_cumsum_vector(
        g, chunk_size=chunk_size, reverse=reverse,
        cu_seqlens=None, head_first=False, output_dtype=torch.float32
    )

    # PyTorch reference
    ref_out = pytorch_local_cumsum_vector(g, chunk_size=chunk_size, reverse=reverse)

    # compare
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-5)
    print("Single random vectorised case PASSED")
