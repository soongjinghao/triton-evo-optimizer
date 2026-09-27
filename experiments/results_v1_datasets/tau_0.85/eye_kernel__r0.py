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
    base_i = pid_i * BLOCK_i
    off_i = base_i + tl.arange(0, BLOCK_i)
    pid_j = tl.program_id(1)
    base_j = pid_j * BLOCK_j
    off_j = base_j + tl.arange(0, BLOCK_j)
    val = (off_i[:, None] == off_j[None, :]).to(tl.float32)
    block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(N, M),
        strides=(M, 1),
        offsets=(base_i, base_j),
        block_shape=(BLOCK_i, BLOCK_j),
        order=(1, 0),
    )
    tl.store(block_ptr, val, boundary_check=(0, 1))
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