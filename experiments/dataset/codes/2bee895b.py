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
    total_elements = N * M
    linear_offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = linear_offset < total_elements
    i = linear_offset // M
    j = linear_offset % M
    val = tl.where(i == j, 1.0, 0.0)
    tl.store(out_ptr + linear_offset, val, mask=mask)

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
    BLOCK = 1024
    grid = (triton.cdiv(total_elements, BLOCK), 1)
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK,
    )
    return out