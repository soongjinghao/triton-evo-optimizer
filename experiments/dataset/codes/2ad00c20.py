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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_flat = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_i = off_flat // M
    off_j = off_flat % M
    mask_i = off_i < N
    mask_j = off_j < M
    mask = mask_i & mask_j
    val = tl.where(off_i == off_j, 1.0, 0.0)
    tl.store(out_ptr + off_flat, val, mask=mask)

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
    total_elements = n * m
    if total_elements >= 4096:
        BLOCK_SIZE = 1024
    elif total_elements >= 1024:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK_SIZE,
    )
    return out