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
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N
    pid_j = tl.program_id(1)
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M
    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i[:, None] * M + off_j[None, :]
    tl.store(out_ptr + off_ij, val, mask=mask)


@triton.jit
def eye_tile_sparse_kernel(
    out_ptr,
    n,
    m,
    BLOCK_i: tl.constexpr,
    BLOCK_j: tl.constexpr,
    BLOCK_DIAG: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    r0 = pid_i * BLOCK_i
    c0 = pid_j * BLOCK_j

    start = tl.maximum(r0, c0)
    end = tl.minimum(
        tl.minimum(r0 + BLOCK_i, c0 + BLOCK_j),
        tl.minimum(n, m),
    )
    length = end - start

    offs = tl.arange(0, BLOCK_DIAG)
    tl.store(out_ptr + (start + offs) * (m + 1), 1.0, mask=offs < length)


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    logger.debug("GEMS EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('npu')
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")

    out = torch.zeros(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )

    if m >= 64:
        BLOCK_i = 16
        BLOCK_j = 64
    else:
        BLOCK_i = 32
        BLOCK_j = 32

    BLOCK_DIAG = min(BLOCK_i, BLOCK_j)
    grid = (triton.cdiv(n, BLOCK_i), triton.cdiv(m, BLOCK_j))

    eye_tile_sparse_kernel[grid](
        out,
        n,
        m,
        BLOCK_i,
        BLOCK_j,
        BLOCK_DIAG,
    )
    return out