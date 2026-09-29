import torch
import triton
import triton.language as tl
@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    if N <= 1024:
        BLOCK_N: tl.constexpr = triton.next_power_of_2(N)
        rindex = tl.arange(0, BLOCK_N)[None, :]
        col_mask = rindex < N
        xs = tl.load(X + (rindex + N * row_idx), xmask & col_mask).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(tl.where(xmask & col_mask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask & col_mask)
    else:
        BLOCK_N: tl.constexpr = 512
        rindex = tl.arange(0, BLOCK_N)[None, :]
        square_sum = tl.zeros([MBLOCK, 1], tl.float32)
        for start in range(0, N, BLOCK_N):
            col_offset = start + rindex
            col_mask = col_offset < N
            xs = tl.load(X + (col_offset + N * row_idx), xmask & col_mask).to(tl.float32)
            square = xs * xs
            square_sum += tl.sum(tl.where(xmask & col_mask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        for start in range(0, N, BLOCK_N):
            col_offset = start + rindex
            col_mask = col_offset < N
            xs = tl.load(X + (col_offset + N * row_idx), xmask & col_mask).to(tl.float32)
            tl.store(Y + (col_offset + N * row_idx), xs * rsqrt, xmask & col_mask)
def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    MBLOCK = 32
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        num_warps=8,
        num_stages=2,
    )
    return y.view(x_shape_og)