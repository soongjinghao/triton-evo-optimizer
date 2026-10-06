import torch
import numpy as np
import os
from collections.abc import Callable
from typing import Any
import torch
import triton
import triton.language as tl

@triton.jit
def _compute_pid_swizzled(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

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
    tile_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid_swizzled(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 循环不变量外提:offs_am[:, None] * stride_am 和 offs_bn[None, :] * stride_bn
    offs_am_strided = offs_am[:, None] * stride_am
    offs_bn_strided = offs_bn[None, :] * stride_bn

    # 循环不变量外提:a_mask 和 b_mask 中与循环无关的部分
    a_mask_row = offs_am[:, None] < M
    b_mask_col = offs_bn[None, :] < N

    # 基地址计算:a_ptr + offs_am_strided,b_ptr + offs_bn_strided
    a_base = a_ptr + offs_am_strided
    b_base = b_ptr + offs_bn_strided

    # 循环内仅累加 BLOCK_SIZE_K * stride_ak/bk
    a_step = BLOCK_SIZE_K * stride_ak
    b_step = BLOCK_SIZE_K * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # k_remaining 计算外提:k_start = ki * BLOCK_SIZE_K
    for ki in range(k_tiles):
        k_start = ki * BLOCK_SIZE_K
        k_remaining = K - k_start
        offs_k_current = offs_k + k_start
        a_mask = a_mask_row & (offs_k_current[None, :] < k_remaining)
        b_mask = (offs_k_current[:, None] < k_remaining) & b_mask_col

        a_ptrs = a_base + offs_k_current[None, :] * stride_ak
        b_ptrs = b_base + offs_k_current[:, None] * stride_bk

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_mask = offs_bias < N
        bias = tl.load(bias_ptr + offs_bias, mask=bias_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
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

    def grid(META):
        return  (triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    configs = {
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
        **configs[dtype],
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