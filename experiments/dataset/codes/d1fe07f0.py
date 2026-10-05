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
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N // group_size, K)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    row_start = pid_m * BLOCK_M
    col_start = pid_n * K * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    num_groups = K
    for k in range(num_groups):
        group_offset = k * BLOCK_N
        cols = group_offset + tl.arange(0, BLOCK_N)
        col_mask = (col_start + group_offset + tl.arange(0, BLOCK_N)) < N
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = X_ptr + rows[:, None] * N + col_start + group_offset + tl.arange(0, BLOCK_N)[None, :]
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
        y_ptrs = Y_ptr + rows[:, None] * N + col_start + group_offset + tl.arange(0, BLOCK_N)[None, :]
        tl.store(y_ptrs, y, mask=mask)
        s_cols = pid_n * K + k
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
    K = 4
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N // block_size, K)
    grid = (grid_m * grid_n,)
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
        K=K,
        num_warps=4,
        num_stages=0 if round_scale else 2,
    )
    return y, s