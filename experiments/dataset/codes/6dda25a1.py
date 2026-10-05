import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    num_programs = tl.num_programs(0)
    pid = tl.program_id(0)
    rows_per_program = tl.cdiv(M, num_programs)
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, M)
    
    for i in range(start_row, end_row, MBLOCK):
        row_offset = i
        row_idx = row_offset + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M
        rindex = tl.arange(0, N)[None, :]
        xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
        square = tl.broadcast_to(xs * xs, [MBLOCK, N])
        square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)

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
    MBLOCK = triton.cdiv(T, 4)
    l2norm_fwd_kernel2[(4,)](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
    )
    return y.view(x_shape_og)