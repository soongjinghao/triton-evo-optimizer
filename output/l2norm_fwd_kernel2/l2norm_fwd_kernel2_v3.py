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
    row_idx = row_start + tl.arange(0, MBLOCK)[:, None]  # (MBLOCK, 1)
    row_mask = row_idx < M  # (MBLOCK, 1)

    # 预计算每行的基地址,避免在循环内重复计算 N * row_idx
    row_base = X + N * row_idx  # (MBLOCK, 1) 基地址,指向各行的第一个元素

    square_sum = tl.zeros((MBLOCK,), dtype=tl.float32)
    for n_start in range(0, N, NBLOCK):
        n_offs = n_start + tl.arange(0, NBLOCK)  # (NBLOCK,)
        n_mask = n_offs < N  # (NBLOCK,)
        load_mask = row_mask & n_mask[None, :]  # (MBLOCK, NBLOCK)
        xs = tl.load(
            row_base + n_offs[None, :],
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)  # (MBLOCK, NBLOCK)
        square = xs * xs
        square_sum += tl.sum(square, axis=1)  # (MBLOCK,)

    rsqrt = tl.rsqrt(square_sum + eps)[:, None]  # (MBLOCK, 1)

    # 写回时也使用预计算的 row_base
    row_base_y = Y + N * row_idx  # 同理预计算输出基地址
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