import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def eye_kernel(
    out_ptr,
    N,
    M,
    BLOCK_i: tl.constexpr,
    BLOCK_j: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M
    
    # Strategy: avoid off_i[:, None] and off_j[None, :] recomputation
    # Use direct arange subtraction to compute identity matrix
    row_idx = tl.arange(0, BLOCK_i)[:, None]
    col_idx = tl.arange(0, BLOCK_j)[None, :]
    val = tl.where(row_idx - col_idx == 0, 1.0, 0.0)
    
    mask = mask_i[:, None] & mask_j[None, :]
    
    # Strategy: compute off_ij directly without off_i/off_j intermediate
    off_ij = (pid_i * BLOCK_i + row_idx) * M + (pid_j * BLOCK_j + col_idx)
    tl.store(out_ptr + off_ij, val, mask=mask)


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    logger.debug("GEMS EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('npu')
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")
    out = torch.empty(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )
    if m >= 64:
        BLOCK_i = 16
        BLOCK_j = 64
    else:
        BLOCK_i = 32
        BLOCK_j = 32
    grid = (triton.cdiv(n, BLOCK_i), triton.cdiv(m, BLOCK_j))
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK_i,
        BLOCK_j,
    )
    return out