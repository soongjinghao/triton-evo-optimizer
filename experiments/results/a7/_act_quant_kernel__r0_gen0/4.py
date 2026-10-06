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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    GROUP_M: tl.constexpr = 16
    if BLOCK_M >= 32 and BLOCK_M % GROUP_M == 0:
        # Group 1: load and compute amax1
        rows1 = row_start + tl.arange(0, GROUP_M)
        row_mask1 = rows1 < M
        mask1 = row_mask1[:, None] & col_mask[None, :]
        x_ptrs1 = X_ptr + rows1[:, None] * N + cols[None, :]
        x1 = tl.load(x_ptrs1, mask=mask1, other=0.0).to(tl.float32)
        x_abs1 = tl.abs(x1)
        amax1 = tl.max(x_abs1, axis=1)
        # Prefetch group 2: load while computing amax1
        rows2 = row_start + GROUP_M + tl.arange(0, GROUP_M)
        row_mask2 = rows2 < M
        mask2 = row_mask2[:, None] & col_mask[None, :]
        x_ptrs2 = X_ptr + rows2[:, None] * N + cols[None, :]
        x2 = tl.load(x_ptrs2, mask=mask2, other=0.0).to(tl.float32)
        x_abs2 = tl.abs(x2)
        amax2 = tl.max(x_abs2, axis=1)
        # Merge amax
        amax = tl.maximum(amax1, amax2)
        # Handle remaining rows if BLOCK_M > 2*GROUP_M
        remaining = BLOCK_M - 2 * GROUP_M
        if remaining > 0:
            rows_rem = row_start + 2 * GROUP_M + tl.arange(0, remaining)
            row_mask_rem = rows_rem < M
            mask_rem = row_mask_rem[:, None] & col_mask[None, :]
            x_ptrs_rem = X_ptr + rows_rem[:, None] * N + cols[None, :]
            x_rem = tl.load(x_ptrs_rem, mask=mask_rem, other=0.0).to(tl.float32)
            x_abs_rem = tl.abs(x_rem)
            amax_rem = tl.max(x_abs_rem, axis=1)
            amax = tl.maximum(amax, amax_rem)
        # Reconstruct full x for store
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_abs = tl.abs(x)
        amax = tl.max(x_abs, axis=1)
    amax = tl.maximum(amax, 1e-4)
    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)

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
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None
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
        num_stages=0 if round_scale else 2,
    )
    return y, s