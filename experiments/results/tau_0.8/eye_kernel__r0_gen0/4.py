import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def eye_kernel(
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_i: tl.constexpr,
    BLOCK_j: tl.constexpr,
):
    pid_i = tl.program_id(0)
    base_i = pid_i * BLOCK_i
    off_i = base_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N

    pid_j = tl.program_id(1)
    base_j = pid_j * BLOCK_j
    off_j = base_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M

    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i[:, None] * M + off_j[None, :]

    out_dtype = out_ptr.dtype.element_ty
    zero_tile = tl.zeros((BLOCK_i, BLOCK_j), dtype=out_dtype)

    # Tile is completely outside the main diagonal.
    if base_i + BLOCK_i <= base_j:
        tl.store(out_ptr + off_ij, zero_tile, mask=mask)
    elif base_j + BLOCK_j <= base_i:
        tl.store(out_ptr + off_ij, zero_tile, mask=mask)
    else:
        diag = off_i[:, None] == off_j[None, :]
        one_tile = tl.full((BLOCK_i, BLOCK_j), 1.0, dtype=out_dtype)
        val = tl.where(diag, one_tile, zero_tile)
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
        BLOCK_i = 32
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