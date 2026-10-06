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
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    s_ptrs = S_ptr + rows * (N // group_size) + pid_n
    s_mask = row_mask
    scale = tl.load(s_ptrs, mask=s_mask, other=1.0)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

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
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    round_scale = scale_fmt is not None
    if round_scale:
        x_abs = x_flat.abs()
        amax_per_row = x_abs.view(M, -1, block_size).amax(dim=-1).clamp(min=1e-4)
        log_val = torch.log2(amax_per_row * fp8_max_inv)
        log_ceil = torch.ceil(log_val)
        scale_pre = torch.exp2(log_ceil)
    else:
        x_abs = x_flat.abs()
        amax_per_row = x_abs.view(M, -1, block_size).amax(dim=-1).clamp(min=1e-4)
        scale_pre = amax_per_row * fp8_max_inv
    s_flat.copy_(scale_pre)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
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