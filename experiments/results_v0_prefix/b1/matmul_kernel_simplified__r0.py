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
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_m_mask = offs_m[:, None] < M
    b_n_mask = offs_n[None, :] < N
    c_mask = a_m_mask & b_n_mask

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    stride_ak_block = BLOCK_SIZE_K * stride_ak
    stride_bk_block = BLOCK_SIZE_K * stride_bk

    for ki in range(k_tiles):
        k_remaining = K - ki * BLOCK_SIZE_K
        a_mask = a_m_mask & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & b_n_mask

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += stride_ak_block
        b_ptrs += stride_bk_block

    if HAS_BIAS:
        bias_mask = offs_n < N
        bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    torch.float32: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
}


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

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    config = _CONFIGS[dtype]
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
        HAS_BIAS=bias is not None,
        **config,
    )
    return c