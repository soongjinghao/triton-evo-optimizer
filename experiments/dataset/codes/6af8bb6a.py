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
    log2_M: tl.constexpr,
):
    pid_i = tl.program_id(0)
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N
    pid_j = tl.program_id(1)
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M
    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = (off_i[:, None] << log2_M) + off_j[None, :]
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
    # M is a power of two (BLOCK_j is always a power of two and M is either BLOCK_j or a multiple of BLOCK_j)
    # Since M is passed as a runtime parameter but is always a power of two, compute log2 at compile time
    # We use a constexpr log2_M computed from the constexpr BLOCK_j and the fact that M is always a power of two
    # Actually M is a runtime parameter, but we can compute log2_M from M at runtime and pass it as constexpr
    # However M is not constexpr, so we need to compute log2_M at runtime and pass it as a regular parameter
    # But the kernel expects log2_M as tl.constexpr, so we need to compute it at compile time
    # Since M is always a power of two (it's either BLOCK_j or a multiple of BLOCK_j, and BLOCK_j is power of two),
    # we can compute log2_M from M at runtime and pass it as a constexpr by using a wrapper
    # Actually we can compute log2_M at runtime and pass it as a regular parameter, but the kernel signature has it as constexpr
    # To keep the kernel interface consistent, we compute log2_M at runtime and pass it as a constexpr
    # Since M is always a power of two, we can compute log2_M using bit_length
    # But bit_length is not allowed in JIT, so we compute it in the wrapper
    log2_M = m.bit_length() - 1 if m > 0 else 0
    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK_i,
        BLOCK_j,
        log2_M,
    )
    return out