import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_simplified(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_cm = offs_am
    offs_cn = offs_bn
    c_mask_m = offs_cm[:, None] < M
    c_mask_n = offs_cn[None, :] < N
    c_mask = c_mask_m & c_mask_n
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    k_remaining = K
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
    b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    if HAS_BIAS:
        offs_bias = offs_bn
        bias_mask = offs_bias < N
        bias = tl.load(bias_ptr + offs_bias, mask=bias_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, (
        "Currently assuming bias is 1D, let Horace know if you run into this"
    )
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 48
    BLOCK_SIZE_K = 64
    grid = (1,)
    matmul_kernel_simplified[grid](
        a,
        b,
        c,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return c

def test_matmul_persistent_float32():
    device = 'npu'
    M, K, N = 32, 64, 48
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    for _ in range(11):
        result = matmul_persistent(a, b, None)
    expected = torch.matmul(a, b)
    assert result.device.type == 'npu', "Output tensor should be on NPU"
    assert result.shape == (M, N), f"Expected shape {(M, N)}, got {result.shape}"
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("? Float32 matmul test passed")

if __name__ == "__main__":
    test_matmul_persistent_float32()
    print("All tests passed!")