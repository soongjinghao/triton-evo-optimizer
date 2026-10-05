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
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    fp8_max_inv: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    s_cols = pid_n
    s_ptrs_base = S_ptr + (N // group_size)
    for i in range(4):
        row = row_start + i
        row_mask = row < M
        mask = row_mask & col_mask
        x_ptrs = X_ptr + row * N + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_abs = tl.abs(x)
        amax = tl.max(x_abs, axis=0)
        amax = tl.maximum(amax, 1e-4)
        if round_scale:
            log_val = tl.log2(amax * fp8_max_inv)
            log_ceil = tl.ceil(log_val)
            scale = tl.exp2(log_ceil)
        else:
            scale = amax * fp8_max_inv
        y = x / scale
        y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
        y_ptrs = Y_ptr + row * N + cols
        tl.store(y_ptrs, y, mask=mask)
        s_ptrs = s_ptrs_base * row + s_cols
        tl.store(s_ptrs, scale, mask=row_mask)

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
    BLOCK_M = 128
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
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
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        fp8_max_inv=fp8_max_inv,
        num_stages=0 if round_scale else 2,
    )
    return y, s