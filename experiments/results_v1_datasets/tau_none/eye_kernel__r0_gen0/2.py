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
    i0 = pid_i * BLOCK_i
    i1 = i0 + BLOCK_i
    off_i = i0 + tl.arange(0, BLOCK_i)
    mask_i = off_i < N

    pid_j = tl.program_id(1)
    j0 = pid_j * BLOCK_j
    j1 = j0 + BLOCK_j
    off_j = j0 + tl.arange(0, BLOCK_j)
    mask_j = off_j < M

    if i1 <= j0:
        val = tl.zeros([BLOCK_i, BLOCK_j], out_ptr.dtype.element_ty)
    elif j1 <= i0:
        val = tl.zeros([BLOCK_i, BLOCK_j], out_ptr.dtype.element_ty)
    else:
        val = tl.cast(
            tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0),
            out_ptr.dtype.element_ty,
        )

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