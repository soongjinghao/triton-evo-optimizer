import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid = tl.program_id(0)
    num_d_blocks = tl.cdiv(D, BD)
    pid_t = pid // num_d_blocks
    pid_d = pid % num_d_blocks
    start_t = pid_t * BLOCK_SIZE_T
    start_d = pid_d * BD
    t_offsets = tl.arange(0, BLOCK_SIZE_T)
    d_offsets = tl.arange(0, BD)
    x_ptrs = x + (start_t + t_offsets[:, None]) * D + (start_d + d_offsets[None, :])
    y_ptrs = y + (start_t + t_offsets[:, None]) * D + (start_d + d_offsets[None, :])
    t_mask = (start_t + t_offsets) < tl.num_programs(0) // num_d_blocks * BLOCK_SIZE_T
    d_mask = (start_d + d_offsets) < D
    mask = t_mask[:, None] & d_mask[None, :]
    b_x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    b_x_squared = b_x * b_x
    b_var = tl.sum(b_x_squared, axis=1)
    b_rstd = tl.rsqrt(b_var + eps)
    b_y = b_x * b_rstd[:, None]
    tl.store(y_ptrs, b_y, mask=mask)

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
    MAX_CORES = 32
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    BLOCK_SIZE_T = min(4, triton.next_power_of_2(T))
    if BLOCK_SIZE_T * D > 2**15:
        BLOCK_SIZE_T = 1
    grid_t = triton.cdiv(T, BLOCK_SIZE_T)
    grid_t = min(grid_t, MAX_CORES)
    if grid_t * BLOCK_SIZE_T < T:
        BLOCK_SIZE_T = triton.next_power_of_2(triton.cdiv(T, grid_t))
    grid_d = triton.cdiv(D, BD)
    grid = (grid_t * grid_d,)
    l2norm_fwd_kernel1[grid](
        x,
        y,
        eps=eps,
        D=D,
        BD=BD,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    return y.view(x_shape_og)