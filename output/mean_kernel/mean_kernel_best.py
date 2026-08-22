import torch
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
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_dim_k = tl.cdiv(K, BLOCK_K)
    m = pid // grid_dim_k
    k_block = pid % grid_dim_k
    if m >= M:
        return
    k_start = k_block * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    if BLOCK_SIZE == 128:
        # 双倍展开循环
        if input_stride2 == 1:
            for n_start in range(0, N, 2 * BLOCK_SIZE):
                n_offsets1 = n_start + tl.arange(0, BLOCK_SIZE)
                mask1 = (n_offsets1[:, None] < N) & k_mask[None, :]
                base1 = input_ptr + m * input_stride0 + n_start * input_stride1
                idx1 = base1 + (tl.arange(0, BLOCK_SIZE) * input_stride1)[:, None] + k_offsets[None, :]
                vals1 = tl.load(idx1, mask=mask1, other=0.0)
                acc += tl.sum(vals1, axis=0)

                n_offsets2 = n_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask2 = (n_offsets2[:, None] < N) & k_mask[None, :]
                base2 = input_ptr + m * input_stride0 + (n_start + BLOCK_SIZE) * input_stride1
                idx2 = base2 + (tl.arange(0, BLOCK_SIZE) * input_stride1)[:, None] + k_offsets[None, :]
                vals2 = tl.load(idx2, mask=mask2, other=0.0)
                acc += tl.sum(vals2, axis=0)
        else:
            for n_start in range(0, N, 2 * BLOCK_SIZE):
                n_offsets1 = n_start + tl.arange(0, BLOCK_SIZE)
                mask1 = (n_offsets1[:, None] < N) & k_mask[None, :]
                input_idx1 = (
                    m * input_stride0
                    + n_offsets1[:, None] * input_stride1
                    + k_offsets[None, :] * input_stride2
                )
                vals1 = tl.load(input_ptr + input_idx1, mask=mask1, other=0.0)
                acc += tl.sum(vals1, axis=0)

                n_offsets2 = n_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask2 = (n_offsets2[:, None] < N) & k_mask[None, :]
                input_idx2 = (
                    m * input_stride0
                    + n_offsets2[:, None] * input_stride1
                    + k_offsets[None, :] * input_stride2
                )
                vals2 = tl.load(input_ptr + input_idx2, mask=mask2, other=0.0)
                acc += tl.sum(vals2, axis=0)
    else:
        if input_stride2 == 1:
            for n_start in range(0, N, BLOCK_SIZE):
                n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
                n_mask = n_offsets < N
                mask = n_mask[:, None] & k_mask[None, :]
                base = input_ptr + m * input_stride0 + n_start * input_stride1
                idx = base + (tl.arange(0, BLOCK_SIZE) * input_stride1)[:, None] + k_offsets[None, :]
                vals = tl.load(idx, mask=mask, other=0.0)
                acc += tl.sum(vals, axis=0)
        else:
            for n_start in range(0, N, BLOCK_SIZE):
                n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
                n_mask = n_offsets < N
                mask = n_mask[:, None] & k_mask[None, :]
                idx = (
                    m * input_stride0
                    + n_offsets[:, None] * input_stride1
                    + k_offsets[None, :] * input_stride2
                )
                vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
                acc += tl.sum(vals, axis=0)

    mean = acc / N
    output_idx = m * output_stride0 + k_offsets * output_stride1
    store_mask = k_offsets < K
    tl.store(output_ptr + output_idx, mean, mask=store_mask)

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

    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    if K >= 256:
        BLOCK_K = 256
        BLOCK_SIZE = 128
    else:
        BLOCK_K = min(K, 64)
        BLOCK_SIZE = min(N, 65536)
        BLOCK_SIZE = max(BLOCK_SIZE, 1)

    grid = (M * ((K + BLOCK_K - 1) // BLOCK_K),)
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
        BLOCK_K,
    )
    return output