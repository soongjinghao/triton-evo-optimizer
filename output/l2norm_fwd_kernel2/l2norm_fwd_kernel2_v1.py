import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(
    X,
    Y,
    eps,
    M,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * MBLOCK
    row_idx = row_start + tl.arange(0, MBLOCK)[:, None]
    row_mask = row_idx < M
    row_base = X + N * row_idx
    square_sum = tl.zeros((MBLOCK, 1), dtype=tl.float32)
    for n_start in range(0, N, NBLOCK):
        n_offs = n_start + tl.arange(0, NBLOCK)
        n_mask = n_offs < N
        load_mask = row_mask & n_mask[None, :]
        xs = tl.load(
            row_base + n_offs[None, :],
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum += tl.sum(xs * xs, axis=1, keep_dims=True)
    rsqrt = tl.rsqrt(square_sum + eps)
    row_base_y = Y + N * row_idx
    for n_start in range(0, N, NBLOCK):
        n_offs = n_start + tl.arange(0, NBLOCK)
        n_mask = n_offs < N
        store_mask = row_mask & n_mask[None, :]
        xs = tl.load(
            row_base + n_offs[None, :],
            mask=store_mask,
            other=0.0,
        ).to(tl.float32)
        out = xs * rsqrt
        tl.store(row_base_y + n_offs[None, :], out, mask=store_mask)

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
    grid = (triton.cdiv(T, MBLOCK),)
    l2norm_fwd_kernel2[grid](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        NBLOCK,
    )
    return y.view(x_shape_og)