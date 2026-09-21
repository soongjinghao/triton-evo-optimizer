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
    PACKED_GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_view = tl.view(x, (BLOCK_M * PACKED_GROUPS, group_size))
    x_abs = tl.abs(x_view)
    amax = tl.max(x_abs, axis=1)
    amax = tl.maximum(amax, 1e-4)

    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    scale_broadcast = scale[:, None]
    y_view = x_view / scale_broadcast
    y_view = tl.minimum(tl.maximum(y_view, fp8_min), fp8_max)
    y = tl.view(y_view, (BLOCK_M, BLOCK_N))

    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    scale_2d = tl.view(scale, (BLOCK_M, PACKED_GROUPS))
    num_groups = N // group_size
    group_ids = pid_n * PACKED_GROUPS + tl.arange(0, PACKED_GROUPS)
    s_ptrs = S_ptr + rows[:, None] * num_groups + group_ids[None, :]
    s_mask = row_mask[:, None] & (group_ids[None, :] < num_groups)
    tl.store(s_ptrs, scale_2d, mask=s_mask)

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
    PACKED_GROUPS = 2
    BLOCK_N = PACKED_GROUPS * block_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

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
        PACKED_GROUPS=PACKED_GROUPS,
        num_stages=0 if round_scale else 2,
    )

    return y, s