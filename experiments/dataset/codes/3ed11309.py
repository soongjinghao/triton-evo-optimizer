import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    total_rows = K * MBLOCK
    row_start = pid * total_rows
    offsets = row_start + tl.arange(0, total_rows)[:, None]
    row_mask = offsets < M
    col_offsets = tl.arange(0, N)[None, :]
    
    xs = tl.load(X + (col_offsets + N * offsets), row_mask).to(tl.float32)
    square = xs * xs
    square_sum = tl.sum(tl.where(row_mask, square, 0.0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (col_offsets + N * offsets), xs * rsqrt, row_mask)

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
    K = 2 if D <= 4096 else 1
    total_rows_per_program = K * MBLOCK
    grid = (triton.cdiv(T, total_rows_per_program),)
    l2norm_fwd_kernel2[grid](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        K,
    )
    return y.view(x_shape_og)