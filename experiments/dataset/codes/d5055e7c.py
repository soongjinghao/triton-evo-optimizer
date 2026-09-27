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
    # Stride elimination: N is power of 2 (group_size * num_groups), use shift
    # N is constexpr from wrapper, group_size is constexpr, num_groups = N // group_size
    # N is always power of 2 because last dim divisible by block_size and block_size is power of 2
    # Use shift: rows[:, None] << log2N + cols[None, :]
    log2N: tl.constexpr = 0
    # Compute log2N at compile time via constexpr trick
    # N is constexpr, we can use a constexpr function
    # Since N is power of 2, log2N = N.bit_length() - 1, but bit_length not allowed in jit
    # Use constexpr from wrapper: pass log2N as constexpr
    # Actually we can compute via tl.constexpr in wrapper, but here we keep original for safety
    # The strategy says: if N is power of 2 and stride is 1, use shift
    # N is constexpr, we can prove it's power of 2 from wrapper contract
    # Use a constexpr branch: if N & (N-1) == 0, use shift, else use multiply
    # Since N is constexpr, this branch is resolved at compile time
    # Compute log2N via constexpr: we need to pass it as constexpr from wrapper
    # For simplicity, keep original multiply but with stride simplification
    # Actually the stride is 1 for contiguous tensor, so we can use rows[:, None] * N + cols[None, :]
    # But N is power of 2, so we can use shift
    # We'll compute log2N via constexpr in the kernel using a constexpr function
    # Since tl.constexpr can't call Python methods, we use a workaround:
    # The wrapper passes log2N as a constexpr parameter
    # But we can't change the kernel signature per rules (must keep same signature)
    # So we keep the original multiply, which is fine for correctness
    # The strategy says "配合G2 Guard确保mask覆盖不变" - we keep mask unchanged
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
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