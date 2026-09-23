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
    pid = tl.program_id(0)
    num_blocks_i = tl.cdiv(N, BLOCK_i)
    pid_i = pid % num_blocks_i
    pid_j = pid // num_blocks_i
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M
    off_i_base = off_i * M
    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i_base[:, None] + off_j[None, :]
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
    num_blocks_i = triton.cdiv(n, BLOCK_i)
    num_blocks_j = triton.cdiv(m, BLOCK_j)
    grid = (num_blocks_i * num_blocks_j,)
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK_i,
        BLOCK_j,
        num_warps=4,
        num_stages=2,
    )
    return out