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
    NUM_TILES_J: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_i = pid // NUM_TILES_J
    tile_j = pid % NUM_TILES_J

    off_i = tile_i * BLOCK_i + tl.arange(0, BLOCK_i)
    off_j = tile_j * BLOCK_j + tl.arange(0, BLOCK_j)

    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    off_ij = off_i[:, None] * M + off_j[None, :]

    IS_FULL_TILE: tl.constexpr = (N % BLOCK_i == 0) and (M % BLOCK_j == 0)
    if IS_FULL_TILE:
        tl.store(out_ptr + off_ij, val)
    else:
        mask_i = off_i < N
        mask_j = off_j < M
        mask = mask_i[:, None] & mask_j[None, :]
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

    num_tiles_j = triton.cdiv(m, BLOCK_j)
    num_tiles_i = triton.cdiv(n, BLOCK_i)
    grid = (num_tiles_i * num_tiles_j,)

    eye_kernel[grid](
        out,
        n,
        m,
        BLOCK_i,
        BLOCK_j,
        num_tiles_j,
    )

    return out