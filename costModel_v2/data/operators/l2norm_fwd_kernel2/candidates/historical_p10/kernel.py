import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NB: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    if NB >= N:
        rindex = tl.arange(0, NB)[None, :]
        xs = tl.load(X + (rindex + N * row_idx), xmask & (rindex < N)).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(tl.where(xmask & (rindex < N), square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask & (rindex < N))
    else:
        rindex = tl.arange(0, NB)[None, :]
        xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
        rindex_rem = tl.arange(0, N - NB)[None, :]
        xs_rem = tl.load(X + (rindex_rem + NB + N * row_idx), xmask).to(tl.float32)
        square_rem = xs_rem * xs_rem
        square_sum += tl.sum(tl.where(xmask, square_rem, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)
        tl.store(Y + (rindex_rem + NB + N * row_idx), xs_rem * rsqrt, xmask)

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
    if D <= 128:
        NB = triton.next_power_of_2(D)
    else:
        NB = BD
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        NB,
    )
    return y.view(x_shape_og)