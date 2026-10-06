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
    
    # 循环不变基址:m_idx * input_stride0 + k_idx * input_stride2
    base_offset = m_idx * input_stride0 + k_idx * input_stride2
    
    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_SIZE
        # 连续化访存:n_start * input_stride1 + tl.arange(0, BLOCK_SIZE) * input_stride1
        n_offsets = n_start * input_stride1 + tl.arange(0, BLOCK_SIZE) * input_stride1
        mask = n_offsets < N * input_stride1  # 注意:mask需要基于原始N,但这里n_offsets已乘以stride1
        # 修正:使用原始n_start和tl.arange构造mask
        n_raw = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_raw < N
        input_idx = base_offset + n_offsets
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
    # BLOCK_SIZE调整为min(N, 1024)以匹配UB容量并减少MTE2请求次数
    BLOCK_SIZE = min(N, 1024)
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