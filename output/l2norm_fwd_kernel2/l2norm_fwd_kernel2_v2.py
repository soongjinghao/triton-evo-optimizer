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
    # 每个 program 处理 MBLOCK 行,在特征维度上以 NBLOCK 为步长循环
    pid = tl.program_id(0)
    row_start = pid * MBLOCK
    row_idx = row_start + tl.arange(0, MBLOCK)[:, None]  # (MBLOCK, 1)
    row_mask = row_idx < M  # (MBLOCK, 1)

    # ---- 第一遍:累加平方和 ----
    square_sum = tl.zeros((MBLOCK,), dtype=tl.float32)
    for n_start in range(0, N, NBLOCK):
        n_offs = n_start + tl.arange(0, NBLOCK)  # (NBLOCK,)
        n_mask = n_offs < N  # (NBLOCK,)
        # 组合行掩码与列掩码,构成 2D mask
        load_mask = row_mask & n_mask[None, :]  # (MBLOCK, NBLOCK)

        # 加载特征块,无效位置用 0 填充,避免显式 where
        xs = tl.load(
            X + n_offs[None, :] + N * row_idx,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)  # (MBLOCK, NBLOCK)

        square = xs * xs
        # 沿特征方向求和,累加
        square_sum += tl.sum(square, axis=1)  # (MBLOCK,)

    # 计算 rsqrt,并加上 eps 防止除零
    rsqrt = tl.rsqrt(square_sum + eps)[:, None]  # (MBLOCK, 1)

    # ---- 第二遍:计算归一化结果并存储 ----
    for n_start in range(0, N, NBLOCK):
        n_offs = n_start + tl.arange(0, NBLOCK)
        n_mask = n_offs < N
        store_mask = row_mask & n_mask[None, :]

        xs = tl.load(
            X + n_offs[None, :] + N * row_idx,
            mask=store_mask,
            other=0.0,
        ).to(tl.float32)

        out = xs * rsqrt
        tl.store(Y + n_offs[None, :] + N * row_idx, out, mask=store_mask)


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

    # 输出最后一维必须连续,保证 store 时步长为 1
    assert y.stride(-1) == 1

    T, D = x.shape[0], x.shape[-1]
    MBLOCK = 32
    NBLOCK = 64

    # 特征分块后不再有单次加载过大 UB 的限制,但保留原接口语义
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