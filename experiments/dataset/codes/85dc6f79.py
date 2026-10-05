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
    off_i = pid * BLOCK // M
    off_j = pid * BLOCK % M
    off_i_arange = off_i + tl.arange(0, BLOCK)
    off_j_arange = off_j + tl.arange(0, BLOCK)
    mask_i = off_i_arange < N
    mask_j = off_j_arange < M
    val = tl.where(off_i_arange[:, None] == off_j_arange[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i_arange[:, None] * M + off_j_arange[None, :]
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
        BLOCK = 1024
    else:
        BLOCK = 1024
    grid = (triton.cdiv(n * m, BLOCK),)
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK,
    )
    return out