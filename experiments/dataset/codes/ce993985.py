import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    sum_sq = tl.zeros([1], dtype=tl.float32)

    # Fast path: use group pack reshape reduce when n_cols is divisible by K * GROUP_SIZE
    fast_block = K * GROUP_SIZE
    if n_cols % fast_block == 0 and input_row_stride == n_cols:
        n_fast_blocks = n_cols // fast_block
        for block_offset in range(0, n_fast_blocks, 1):
            col_offset = block_offset * fast_block
            col_idx = col_offset + tl.arange(0, fast_block)
            vals = tl.load(row_start_ptr + col_idx, mask=None)
            vals_f32 = vals.to(tl.float32)
            # Reshape to (K, GROUP_SIZE) and reduce along axis=1
            vals_2d = tl.view(vals_f32, (K, GROUP_SIZE))
            sq_vals = vals_2d * vals_2d
            sum_sq += tl.sum(tl.sum(sq_vals, axis=1))
        # Serial fallback for remaining columns (none in this case, but kept for safety)
        remaining = 0
    else:
        # Fallback: use original serial loop
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Write-back: use fast path if applicable
    if n_cols % fast_block == 0 and input_row_stride == n_cols:
        n_fast_blocks = n_cols // fast_block
        for block_offset in range(0, n_fast_blocks, 1):
            col_offset = block_offset * fast_block
            col_idx = col_offset + tl.arange(0, fast_block)
            vals = tl.load(row_start_ptr + col_idx, mask=None)
            weight = tl.load(weight_ptr + col_idx, mask=None)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms * weight_f32
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr + col_idx, output, mask=None)
    else:
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms * weight_f32
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr + col_idx, output, mask=mask)

def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    GROUP_SIZE = 256
    K = 4
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
        K=K,
    )
    return output.reshape(original_shape)

def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return rms_norm(input, weight, eps=eps)