import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def l2norm_fwd_kernel1(
    x_t,
    y_t,
    D,
    T,
    BD: tl.constexpr,
    eps,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    start_t = pid_t * BLOCK_SIZE_T
    start_d = pid_d * BD
    
    offs_t = start_t + tl.arange(0, BLOCK_SIZE_T)
    offs_d = start_d + tl.arange(0, BD)
    
    t_mask = offs_t[None, :] < T
    d_mask = offs_d[:, None] < D
    
    b_x = tl.load(x_t + offs_d[:, None] * T + offs_t[None, :], mask=(t_mask & d_mask), other=0.0)
    b_x = b_x.to(tl.float32)
    
    b_x_squared = b_x * b_x
    b_var = tl.sum(b_x_squared, axis=0)
    b_rstd = tl.rsqrt(b_var + eps)
    b_rstd_broadcast = tl.broadcast_to(b_rstd[None, :], (BD, BLOCK_SIZE_T))
    b_y = b_x * b_rstd_broadcast
    
    tl.store(y_t + offs_d[:, None] * T + offs_t[None, :], b_y, mask=(t_mask & d_mask))

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
    
    x_t = x.T.contiguous()
    y_t = torch.empty_like(x_t)
    
    l2norm_fwd_kernel1[(grid_t, grid_d)](
        x_t,
        y_t,
        eps=eps,
        D=D,
        T=T,
        BD=BD,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    
    y = y_t.T.contiguous()
    return y.view(x_shape_og)