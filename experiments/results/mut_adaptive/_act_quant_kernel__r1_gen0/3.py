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
    stride_n: tl.constexpr,
    stride_s: tl.constexpr,
    log2_N: tl.constexpr,
    log2_group_size: tl.constexpr,
    use_bitwise: tl.constexpr,
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
    if use_bitwise:
        x_ptrs = X_ptr + (rows[:, None] << log2_N) + cols[None, :]
    else:
        x_ptrs = X_ptr + rows[:, None] * stride_n + cols[None, :]
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
    if use_bitwise:
        y_ptrs = Y_ptr + (rows[:, None] << log2_N) + cols[None, :]
    else:
        y_ptrs = Y_ptr + rows[:, None] * stride_n + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)
    s_cols = pid_n
    if use_bitwise:
        s_ptrs = S_ptr + (rows << log2_stride_s) + s_cols
    else:
        s_ptrs = S_ptr + rows * stride_s + s_cols
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
    stride_n = N
    stride_s = N // block_size
    use_bitwise = (N & (N - 1)) == 0 and (block_size & (block_size - 1)) == 0
    if use_bitwise:
        log2_N = N.bit_length() - 1
        log2_group_size = block_size.bit_length() - 1
        log2_stride_s = stride_s.bit_length() - 1
    else:
        log2_N = 0
        log2_group_size = 0
        log2_stride_s = 0
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
        stride_n=stride_n,
        stride_s=stride_s,
        log2_N=log2_N,
        log2_group_size=log2_group_size,
        use_bitwise=use_bitwise,
        num_stages=0 if round_scale else 2,
    )
    return y, s