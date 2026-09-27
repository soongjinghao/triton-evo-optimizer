import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _should_use_diagonal_only(n, m, block_i, block_j):
    if n == 0 or m == 0:
        return True

    min_dim = min(n, m)
    area = n * m
    tile_space_writes = (
        triton.cdiv(n, block_i) * block_i *
        triton.cdiv(m, block_j) * block_j
    )

    # Use zero-fill + diagonal-store only when the diagonal band is much
    # smaller than the full matrix area and pre-zeroing does not write more
    # than the original tile-space store footprint.
    return min_dim * 32 <= area and area <= tile_space_writes


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
def eye_kernel_diag(
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

    diag = off_i[:, None] == off_j[None, :]
    val = tl.where(diag, 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :] & diag
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

    if m >= 64:
        BLOCK_i = 16
        BLOCK_j = 64
    else:
        BLOCK_i = 32
        BLOCK_j = 32

    grid = (triton.cdiv(n, BLOCK_i), triton.cdiv(m, BLOCK_j))

    if _should_use_diagonal_only(n, m, BLOCK_i, BLOCK_j):
        out = torch.zeros(
            (n, m),
            dtype=dtype,
            device=device,
            layout=layout,
            pin_memory=pin_memory,
        )
        eye_kernel_diag[grid](
            out,
            n,
            m,
            BLOCK_i,
            BLOCK_j,
        )
    else:
        out = torch.empty(
            (n, m),
            dtype=dtype,
            device=device,
            layout=layout,
            pin_memory=pin_memory,
        )
        eye_kernel[grid](
            out,
            n,
            m,
            BLOCK_i,
            BLOCK_j,
        )

    return out