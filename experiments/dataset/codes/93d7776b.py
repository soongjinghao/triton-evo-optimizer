import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    base_offset = N * row_idx
    xs = tl.load(X + (rindex + base_offset), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + base_offset), xs * rsqrt, xmask)

@triton.jit
def l2norm_fwd_kernel2_grouped(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_groups = tl.cdiv(M, MBLOCK * GROUP_SIZE)
    group_id = pid // num_groups
    group_start = group_id * MBLOCK * GROUP_SIZE
    local_id = pid % num_groups
    xoffset = group_start + local_id * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    base_offset = N * row_idx
    xs = tl.load(X + (rindex + base_offset), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + base_offset), xs * rsqrt, xmask)

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
    if T < 16:
        MBLOCK = 32
    else:
        MBLOCK = 16
    GROUP_SIZE = 4
    num_groups = triton.cdiv(T, MBLOCK * GROUP_SIZE)
    grid = (num_groups * GROUP_SIZE,)
    l2norm_fwd_kernel2_grouped[grid](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
        GROUP_SIZE,
    )
    return y.view(x_shape_og)