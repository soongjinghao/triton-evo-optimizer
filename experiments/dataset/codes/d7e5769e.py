import torch
import torch_npu
import triton
import triton.language as tl
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid // K
    k_idx = pid % K
    if m_idx >= M or k_idx >= K:
        return
    acc = 0.0
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_SIZE
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)
@triton.jit
def mean_kernel_dual_m(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
    M_EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    k_idx = pid % K
    m_base = (pid // K) * 2
    m_idx0 = m_base
    m_idx1 = m_base + 1
    if m_idx0 >= M or k_idx >= K:
        return
    acc0 = 0.0
    acc1 = 0.0
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_SIZE
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        input_idx0 = (
            m_idx0 * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        vals0 = tl.load(input_ptr + input_idx0, mask=mask, other=0.0)
        acc0 += tl.sum(vals0)
        if M_EVEN:
            input_idx1 = (
                m_idx1 * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
            )
            vals1 = tl.load(input_ptr + input_idx1, mask=mask, other=0.0)
            acc1 += tl.sum(vals1)
    mean_val0 = acc0 / N
    output_idx0 = m_idx0 * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx0, mean_val0)
    if M_EVEN:
        mean_val1 = acc1 / N
        output_idx1 = m_idx1 * output_stride0 + k_idx * output_stride1
        tl.store(output_ptr + output_idx1, mean_val1)
def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )
    if dim < 0:
        dim = dim + input.ndim
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype
    if input.dtype != dtype:
        input = input.to(dtype)
    shape = list(input.shape)
    M = 1
    for i in range(dim):
        M *= shape[i]
    N = shape[dim]
    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]
    input_3d = input.reshape(M, N, K)
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]
    output = torch.empty(output_shape, dtype=dtype, device=input.device)
    output_2d = output.reshape(M, 1, K).squeeze(1) if keepdim else output.reshape(M, K)
    MAX_BLOCKS_UB = 65536
    BLOCK_SIZE = min(N, MAX_BLOCKS_UB)
    BLOCK_SIZE = max(BLOCK_SIZE, 1)
    if M % 2 == 0 and M >= 2:
        grid = ((M // 2) * K,)
        mean_kernel_dual_m[grid](
            input_3d,
            output_2d,
            input_3d.stride(0),
            input_3d.stride(1),
            input_3d.stride(2),
            output_2d.stride(0),
            output_2d.stride(1) if output_2d.ndim > 1 else 0,
            M,
            N,
            K,
            BLOCK_SIZE,
            True,
            num_warps=8,
            num_stages=2,
        )
    else:
        grid = (M * K,)
        mean_kernel[grid](
            input_3d,
            output_2d,
            input_3d.stride(0),
            input_3d.stride(1),
            input_3d.stride(2),
            output_2d.stride(0),
            output_2d.stride(1) if output_2d.ndim > 1 else 0,
            M,
            N,
            K,
            BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    return output