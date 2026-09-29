import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_rows_per_program = MBLOCK
    row_start = pid * num_rows_per_program
    row_offset = tl.arange(0, num_rows_per_program)[:, None]
    row_idx = row_start + row_offset
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = xs * xs
    square_sum = tl.sum(tl.where(xmask, square, 0.0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)

@triton.jit
def l2norm_fwd_kernel2_packed(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    num_rows_per_program = MBLOCK * BLOCK_M
    row_start = pid * num_rows_per_program
    m_offset = tl.arange(0, BLOCK_M)[:, None]
    row_offset = tl.arange(0, MBLOCK)[None, :]
    row_idx = row_start + m_offset * MBLOCK + row_offset
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = xs * xs
    square_sum = tl.sum(tl.where(xmask, square, 0.0), 1)[:, None]
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
    MBLOCK = 32
    BLOCK_M = 4
    num_packed_rows = MBLOCK * BLOCK_M
    grid = (triton.cdiv(T, num_packed_rows),)
    l2norm_fwd_kernel2_packed[grid](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        BLOCK_M,
    )
    return y.view(x_shape_og)