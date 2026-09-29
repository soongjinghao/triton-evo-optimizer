import torch
import triton
import triton.language as tl
@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)
@triton.jit
def l2norm_fwd_kernel2_large(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, BLOCK_N: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    square_sum = tl.zeros([MBLOCK, 1], tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)[None, :]
        nmask = n_offset < N
        xs = tl.load(X + (n_offset + N * row_idx), xmask & nmask).to(tl.float32)
        square = xs * xs
        square_sum += tl.sum(tl.where(xmask & nmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)[None, :]
        nmask = n_offset < N
        xs = tl.load(X + (n_offset + N * row_idx), xmask & nmask).to(tl.float32)
        tl.store(Y + (n_offset + N * row_idx), xs * rsqrt, xmask & nmask)
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
    if D <= 1024:
        MBLOCK = 64
        l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            MBLOCK,
        )
    else:
        MBLOCK = 32
        BLOCK_N = 1024
        l2norm_fwd_kernel2_large[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            MBLOCK,
            BLOCK_N,
        )
    return y.view(x_shape_og)