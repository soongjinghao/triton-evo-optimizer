import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel1(
    X,
    square_sum_buf,
    M,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * MBLOCK
    row_idx = row_start + tl.arange(0, MBLOCK)
    row_mask = row_idx < M
    col_start = pid_n * NBLOCK
    col_offs = col_start + tl.arange(0, NBLOCK)
    col_mask = col_offs < N
    load_mask = row_mask[:, None] & col_mask[None, :]
    xs = tl.load(
        X + N * row_idx[:, None] + col_offs[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)
    square = xs * xs
    square_sum = tl.sum(square, axis=1)
    for i in range(MBLOCK):
        if row_mask[i]:
            tl.atomic_add(square_sum_buf + row_idx[i], square_sum[i])

@triton.jit
def l2norm_fwd_kernel2(
    X,
    square_sum_buf,
    Y,
    eps,
    M,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * MBLOCK
    row_idx = row_start + tl.arange(0, MBLOCK)
    row_mask = row_idx < M
    col_start = pid_n * NBLOCK
    col_offs = col_start + tl.arange(0, NBLOCK)
    col_mask = col_offs < N
    load_mask = row_mask[:, None] & col_mask[None, :]
    row_sq = tl.load(square_sum_buf + row_idx, mask=row_mask, other=0.0)
    rsqrt = tl.rsqrt(row_sq + eps)
    xs = tl.load(
        X + N * row_idx[:, None] + col_offs[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)
    out = xs * rsqrt[:, None]
    tl.store(
        Y + N * row_idx[:, None] + col_offs[None, :],
        out,
        mask=load_mask,
    )

def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MBLOCK = 32
    NBLOCK = 64
    square_sum_buf = torch.zeros(T, dtype=torch.float32, device=x.device)
    grid_m = triton.cdiv(T, MBLOCK)
    grid_n = triton.cdiv(D, NBLOCK)
    grid = (grid_m, grid_n)
    l2norm_fwd_kernel1[grid](
        x,
        square_sum_buf,
        T,
        D,
        MBLOCK,
        NBLOCK,
    )
    l2norm_fwd_kernel2[grid](
        x,
        square_sum_buf,
        y,
        eps,
        T,
        D,
        MBLOCK,
        NBLOCK,
    )
    return y.view(x_shape_og)