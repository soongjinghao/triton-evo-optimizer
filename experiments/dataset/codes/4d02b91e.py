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
    
    inv_N = 1.0 / N
    acc = 0.0
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    
    for block_idx in range(0, num_blocks, 2):
        n_start0 = block_idx * BLOCK_SIZE
        n_offsets0 = n_start0 + tl.arange(0, BLOCK_SIZE)
        mask0 = n_offsets0 < N
        input_idx0 = (
            m_idx * input_stride0 + n_offsets0 * input_stride1 + k_idx * input_stride2
        )
        vals0 = tl.load(input_ptr + input_idx0, mask=mask0, other=0.0)
        acc += tl.reduce(vals0, axis=0, combine_fn=lambda a, b: a + b)
        
        n_start1 = (block_idx + 1) * BLOCK_SIZE
        n_offsets1 = n_start1 + tl.arange(0, BLOCK_SIZE)
        mask1 = n_offsets1 < N
        input_idx1 = (
            m_idx * input_stride0 + n_offsets1 * input_stride1 + k_idx * input_stride2
        )
        vals1 = tl.load(input_ptr + input_idx1, mask=mask1, other=0.0)
        acc += tl.reduce(vals1, axis=0, combine_fn=lambda a, b: a + b)
    
    mean_val = acc * inv_N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)

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
    grid = (M * K,)
    MAX_BLOCKS_UB = 65536
    BLOCK_SIZE = min(N, MAX_BLOCKS_UB)
    BLOCK_SIZE = max(BLOCK_SIZE, 1)
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
    )
    return output

if __name__ == "__main__":
    input_2d = torch.randn(4096, 8192, device='npu', dtype=torch.float32)
    for _ in range(11):
        result_triton = mean_dim(input_2d, dim=1, keepdim=False)