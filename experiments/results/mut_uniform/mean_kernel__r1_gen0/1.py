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
    
    # Use double buffering only when N >= 2 * BLOCK_SIZE
    if num_blocks >= 2:
        # Prefetch first block
        n_start0 = 0
        n_offsets0 = tl.arange(0, BLOCK_SIZE)
        mask0 = n_offsets0 < N
        input_idx0 = (
            m_idx * input_stride0 + n_offsets0 * input_stride1 + k_idx * input_stride2
        )
        vals0 = tl.load(input_ptr + input_idx0, mask=mask0, other=0.0)
        
        for block_idx in range(1, num_blocks):
            # Prefetch next block
            n_start_next = block_idx * BLOCK_SIZE
            n_offsets_next = n_start_next + tl.arange(0, BLOCK_SIZE)
            mask_next = n_offsets_next < N
            input_idx_next = (
                m_idx * input_stride0 + n_offsets_next * input_stride1 + k_idx * input_stride2
            )
            vals_next = tl.load(input_ptr + input_idx_next, mask=mask_next, other=0.0)
            
            # Compute current block
            acc += tl.sum(vals0)
            
            # Swap buffers
            vals0 = vals_next
        
        # Compute last block
        acc += tl.sum(vals0)
    else:
        # Single buffer fallback for small N
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