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
    IS_DIVISIBLE: tl.constexpr,
):
    row_base = tl.program_id(0).to(tl.int64) * 4
    row_idx = row_base + tl.arange(0, 4)
    sum_sq = tl.zeros([4], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        if IS_DIVISIBLE:
            row_start_ptr = input_ptr + row_idx[:, None] * input_row_stride + col_idx[None, :]
            vals = tl.load(row_start_ptr, mask=None, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(sq_vals, axis=1)
        else:
            mask = col_idx < n_cols
            row_start_ptr = input_ptr + row_idx[:, None] * input_row_stride + col_idx[None, :]
            vals = tl.load(row_start_ptr, mask=mask[None, :], other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(tl.where(mask[None, :], sq_vals, 0.0), axis=1)
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        if IS_DIVISIBLE:
            row_start_ptr = input_ptr + row_idx[:, None] * input_row_stride + col_idx[None, :]
            output_row_start_ptr = output_ptr + row_idx[:, None] * output_row_stride + col_idx[None, :]
            vals = tl.load(row_start_ptr, mask=None, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=None, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms[:, None] * weight_f32[None, :]
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr, output, mask=None)
        else:
            mask = col_idx < n_cols
            row_start_ptr = input_ptr + row_idx[:, None] * input_row_stride + col_idx[None, :]
            output_row_start_ptr = output_ptr + row_idx[:, None] * output_row_stride + col_idx[None, :]
            vals = tl.load(row_start_ptr, mask=mask[None, :], other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms[:, None] * weight_f32[None, :]
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr, output, mask=mask[None, :])

@triton.jit
def _rms_norm_kernel_tail(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        if IS_DIVISIBLE:
            vals = tl.load(row_start_ptr + col_idx, mask=None, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(sq_vals)
        else:
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        if IS_DIVISIBLE:
            vals = tl.load(row_start_ptr + col_idx, mask=None, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=None, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms * weight_f32
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr + col_idx, output, mask=None)
        else:
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
    is_divisible = (n_cols % BLOCK_SIZE == 0)
    n_full_groups = n_rows // 4
    if n_full_groups > 0:
        grid = (n_full_groups,)
        _rms_norm_kernel[grid](
            input_2d,
            weight,
            output,
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_DIVISIBLE=is_divisible,
        )
    tail_start = n_full_groups * 4
    tail_rows = n_rows - tail_start
    if tail_rows > 0:
        grid_tail = (tail_rows,)
        _rms_norm_kernel_tail[grid_tail](
            input_2d[tail_start:],
            weight,
            output[tail_start:],
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_DIVISIBLE=is_divisible,
        )
    return output.reshape(original_shape)

def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return rms_norm(input, weight, eps=eps)