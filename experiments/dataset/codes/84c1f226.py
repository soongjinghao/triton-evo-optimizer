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
    pid_m = tl.program_id(0)
    pid_n_group = tl.program_id(1)
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    n_groups_per_program = K
    group_start = pid_n_group * n_groups_per_program * group_size
    cols_offsets = tl.arange(0, BLOCK_N)
    x_ptrs = X_ptr + rows[:, None] * N + group_start + cols_offsets[None, :]
    col_mask = (group_start + cols_offsets) < N
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_2d = tl.view(x, (BLOCK_M * n_groups_per_program, group_size))
    x_abs = tl.abs(x_2d)
    amax = tl.max(x_abs, axis=1)
    amax = tl.maximum(amax, 1e-4)
    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv
    scale_broadcast = scale[:, None]
    y_2d = x_2d / scale_broadcast
    y_2d = tl.minimum(tl.maximum(y_2d, fp8_min), fp8_max)
    y = tl.view(y_2d, (BLOCK_M, n_groups_per_program * group_size))
    y_ptrs = Y_ptr + rows[:, None] * N + group_start + cols_offsets[None, :]
    tl.store(y_ptrs, y, mask=mask)
    s_base = S_ptr + rows * (N // group_size)
    for g in range(n_groups_per_program):
        s_col = pid_n_group * n_groups_per_program + g
        s_ptrs = s_base + s_col
        s_mask = row_mask & (s_col < (N // group_size))
        tl.store(s_ptrs, scale[g * BLOCK_M:(g + 1) * BLOCK_M], mask=s_mask)

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
    n_groups = N // block_size
    if n_groups < K:
        grid = (triton.cdiv(M, BLOCK_M), n_groups)
        K_fallback = 1
        _act_quant_kernel[grid](
            x_flat,
            y_flat,
            s_flat,
            M,
            N,
            group_size=block_size,
            round_scale=scale_fmt is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            K=K_fallback,
            num_stages=0 if scale_fmt is not None else 2,
        )
    else:
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(n_groups, K))
        _act_quant_kernel[grid](
            x_flat,
            y_flat,
            s_flat,
            M,
            N,
            group_size=block_size,
            round_scale=scale_fmt is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            K=K,
            num_stages=0 if scale_fmt is not None else 2,
        )
    return y, s