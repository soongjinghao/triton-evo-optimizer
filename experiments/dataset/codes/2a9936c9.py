import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    
    N_HALF = N // 2
    rindex0 = tl.arange(0, N_HALF)[None, :]
    rindex1 = N_HALF + tl.arange(0, N_HALF)[None, :]
    
    xs0 = tl.load(X + (rindex0 + N * row_idx), xmask).to(tl.float32)
    xs1 = tl.load(X + (rindex1 + N * row_idx), xmask).to(tl.float32)
    
    square0 = xs0 * xs0
    square1 = xs1 * xs1
    
    square_sum0 = tl.sum(tl.where(xmask, square0, 0.0), 1)[:, None]
    square_sum1 = tl.sum(tl.where(xmask, square1, 0.0), 1)[:, None]
    square_sum = square_sum0 + square_sum1
    
    rsqrt = tl.rsqrt(square_sum + eps)
    
    tl.store(Y + (rindex0 + N * row_idx), xs0 * rsqrt, xmask)
    tl.store(Y + (rindex1 + N * row_idx), xs1 * rsqrt, xmask)

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
    MBLOCK = 64
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
    )
    return y.view(x_shape_og)