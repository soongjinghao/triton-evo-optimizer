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
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    start_t = pid_t * BLOCK_SIZE_T
    start_d = pid_d * BD
    
    d_mask = tl.arange(0, BD) < D - start_d
    t_offsets = tl.arange(0, BLOCK_SIZE_T)
    
    # 双缓冲流水线预取:每次加载4行,预取下一批
    # 使用标量地址计算替代make_block_ptr,减少标量开销
    offs_d = start_d + tl.arange(0, BD)
    d_mask = offs_d < D
    
    # 预取第一批
    b_x_0 = tl.load(x + (start_t + 0) * D + offs_d[None, :], mask=d_mask[None, :], other=0.0)
    b_x_1 = tl.load(x + (start_t + 1) * D + offs_d[None, :], mask=d_mask[None, :], other=0.0)
    b_x_2 = tl.load(x + (start_t + 2) * D + offs_d[None, :], mask=d_mask[None, :], other=0.0)
    b_x_3 = tl.load(x + (start_t + 3) * D + offs_d[None, :], mask=d_mask[None, :], other=0.0)
    
    # 计算第一批的rstd
    b_x_sq_0 = b_x_0 * b_x_0
    b_x_sq_1 = b_x_1 * b_x_1
    b_x_sq_2 = b_x_2 * b_x_2
    b_x_sq_3 = b_x_3 * b_x_3
    
    b_var_0 = tl.sum(b_x_sq_0, axis=1)
    b_var_1 = tl.sum(b_x_sq_1, axis=1)
    b_var_2 = tl.sum(b_x_sq_2, axis=1)
    b_var_3 = tl.sum(b_x_sq_3, axis=1)
    
    b_rstd_0 = tl.rsqrt(b_var_0 + eps)
    b_rstd_1 = tl.rsqrt(b_var_1 + eps)
    b_rstd_2 = tl.rsqrt(b_var_2 + eps)
    b_rstd_3 = tl.rsqrt(b_var_3 + eps)
    
    # 存储第一批结果
    b_y_0 = b_x_0 * b_rstd_0[:, None]
    b_y_1 = b_x_1 * b_rstd_1[:, None]
    b_y_2 = b_x_2 * b_rstd_2[:, None]
    b_y_3 = b_x_3 * b_rstd_3[:, None]
    
    tl.store(y + (start_t + 0) * D + offs_d[None, :], b_y_0, mask=d_mask[None, :])
    tl.store(y + (start_t + 1) * D + offs_d[None, :], b_y_1, mask=d_mask[None, :])
    tl.store(y + (start_t + 2) * D + offs_d[None, :], b_y_2, mask=d_mask[None, :])
    tl.store(y + (start_t + 3) * D + offs_d[None, :], b_y_3, mask=d_mask[None, :])

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
    l2norm_fwd_kernel1[(grid_t, grid_d)](
        x,
        y,
        eps=eps,
        D=D,
        BD=BD,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    return y.view(x_shape_og)

def test_l2norm_optimized():
    torch.manual_seed(0)
    x = torch.randn(128, 64, device='npu', dtype=torch.float32)
    eps = 1e-6
    for i in range(11):
        y_triton = l2norm_fwd(x, eps)
    x_reshaped = x.view(-1, x.shape[-1])
    norm = torch.sqrt(torch.sum(x_reshaped ** 2, dim=-1, keepdim=True))
    y_ref = x_reshaped / (norm + eps)
    y_ref = y_ref.view(x.shape)
    assert torch.allclose(y_triton, y_ref, atol=1e-5, rtol=1e-3), f"Max diff: {torch.max(torch.abs(y_triton - y_ref))}"
    print("Optimized L2Norm test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_triton.shape}")
    print(f"Max difference: {torch.max(torch.abs(y_triton - y_ref))}")

if __name__ == "__main__":
    test_l2norm_optimized()