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
    GRID_j: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_i = pid // GRID_j
    pid_j = pid % GRID_j
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M
    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i[:, None] * M + off_j[None, :]
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
    if n < 32 or m < 32:
        BLOCK_i = 16
        BLOCK_j = 64 if m >= 64 else 32
        grid = (triton.cdiv(n, BLOCK_i), triton.cdiv(m, BLOCK_j))
        eye_kernel[grid](
            out,
            n,
            m,
            BLOCK_i,
            BLOCK_j,
            triton.cdiv(m, BLOCK_j),
        )
    else:
        BLOCK_i = 32
        BLOCK_j = 64
        grid_j = triton.cdiv(m, BLOCK_j)
        grid = (triton.cdiv(n, BLOCK_i) * grid_j,)
        eye_kernel[grid](
            out,
            n,
            m,
            BLOCK_i,
            BLOCK_j,
            grid_j,
        )
    return out