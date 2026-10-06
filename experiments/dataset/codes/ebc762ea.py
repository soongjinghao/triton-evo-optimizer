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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off_flat = pid * BLOCK + tl.arange(0, BLOCK)
    off_i = off_flat // M
    off_j = off_flat % M
    mask = (off_i < N) & (off_j < M)
    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    off_ij = off_i * M + off_j
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
    BLOCK = 1024
    grid = (triton.cdiv(n * m, BLOCK),)
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK,
    )
    return out