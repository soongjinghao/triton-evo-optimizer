from typing import Optional, Tuple
import torch
import triton
import triton.language as tl

@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_group_blocks,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 1)
    K: tl.constexpr = BLOCK_N // group_size
    row = tl.program_id(0)
    gid = tl.program_id(1)
    col_start = gid * group_size * K
    total_groups = N // group_size
    offs = tl.arange(0, BLOCK_N)
    cols = col_start + offs
    col_mask = cols < N
    base = X_ptr + row * N + col_start
    x = tl.load(tl.multiple_of(base, 16) + offs, mask=col_mask, other=0.0).to(tl.float32)
    x_2d = tl.view(x, (K, group_size))
    amax = tl.max(tl.abs(x_2d), axis=1)
    amax = tl.maximum(amax, 1e-4)
    fp8_max_inv = 1.0 / fp8_max
    if round_scale:
        log_val = tl.ceil(tl.log2(amax * fp8_max_inv))
        scale = tl.exp2(log_val)
        inv_scale = tl.exp2(-log_val)
    else:
        scale = amax * fp8_max_inv
        inv_scale = 1.0 / scale
    y_2d = x_2d * inv_scale[:, None]
    y_2d = tl.minimum(tl.maximum(y_2d, fp8_min), fp8_max)
    y = tl.view(y_2d, (BLOCK_N,))
    tl.store(tl.multiple_of(Y_ptr + row * N + col_start, 16) + offs, y, mask=col_mask)
    s_cols = (col_start // group_size) + tl.arange(0, K)
    s_mask = s_cols < total_groups
    tl.store(tl.multiple_of(S_ptr + row * total_groups + (col_start // group_size), 16) + tl.arange(0, K), scale, mask=s_mask)

def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    y = torch.empty_like(x, dtype=torch.float16)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)
    total_groups = N // block_size
    MAX_BLOCK_N = 4096
    max_K = 32
    K = min(total_groups, MAX_BLOCK_N // block_size, max_K)
    K = max(K, 1)
    BLOCK_M = 1
    BLOCK_N = block_size * K
    num_group_blocks = triton.cdiv(total_groups, K)
    grid = (M, num_group_blocks)
    round_scale = scale_fmt is not None
    fp8_min = -448.0
    fp8_max = 448.0
    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_group_blocks=num_group_blocks,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        num_stages=0 if round_scale else 2,
    )
    return y, s