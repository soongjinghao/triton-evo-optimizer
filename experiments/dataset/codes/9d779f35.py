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
):
    pid = tl.program_id(0).to(tl.int64)
    n_rows_per_pid: tl.constexpr = 4
    row_idx_base = pid * n_rows_per_pid
    sum_sq_0 = tl.zeros([1], dtype=tl.float32)
    sum_sq_1 = tl.zeros([1], dtype=tl.float32)
    sum_sq_2 = tl.zeros([1], dtype=tl.float32)
    sum_sq_3 = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals_0 = tl.load(input_ptr + (row_idx_base + 0) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_1 = tl.load(input_ptr + (row_idx_base + 1) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_2 = tl.load(input_ptr + (row_idx_base + 2) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_3 = tl.load(input_ptr + (row_idx_base + 3) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_f32_0 = vals_0.to(tl.float32)
        vals_f32_1 = vals_1.to(tl.float32)
        vals_f32_2 = vals_2.to(tl.float32)
        vals_f32_3 = vals_3.to(tl.float32)
        sq_0 = vals_f32_0 * vals_f32_0
        sq_1 = vals_f32_1 * vals_f32_1
        sq_2 = vals_f32_2 * vals_f32_2
        sq_3 = vals_f32_3 * vals_f32_3
        sum_sq_0 += tl.sum(tl.where(mask, sq_0, 0.0))
        sum_sq_1 += tl.sum(tl.where(mask, sq_1, 0.0))
        sum_sq_2 += tl.sum(tl.where(mask, sq_2, 0.0))
        sum_sq_3 += tl.sum(tl.where(mask, sq_3, 0.0))
    inv_n_cols = 1.0 / n_cols
    mean_sq_0 = sum_sq_0 * inv_n_cols
    mean_sq_1 = sum_sq_1 * inv_n_cols
    mean_sq_2 = sum_sq_2 * inv_n_cols
    mean_sq_3 = sum_sq_3 * inv_n_cols
    rms_0 = tl.sqrt(mean_sq_0 + eps)
    rms_1 = tl.sqrt(mean_sq_1 + eps)
    rms_2 = tl.sqrt(mean_sq_2 + eps)
    rms_3 = tl.sqrt(mean_sq_3 + eps)
    inv_rms_0 = 1.0 / rms_0
    inv_rms_1 = 1.0 / rms_1
    inv_rms_2 = 1.0 / rms_2
    inv_rms_3 = 1.0 / rms_3
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        weight_f32 = weight.to(tl.float32)
        vals_0 = tl.load(input_ptr + (row_idx_base + 0) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_1 = tl.load(input_ptr + (row_idx_base + 1) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_2 = tl.load(input_ptr + (row_idx_base + 2) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_3 = tl.load(input_ptr + (row_idx_base + 3) * input_row_stride + col_idx, mask=mask, other=0.0)
        vals_f32_0 = vals_0.to(tl.float32)
        vals_f32_1 = vals_1.to(tl.float32)
        vals_f32_2 = vals_2.to(tl.float32)
        vals_f32_3 = vals_3.to(tl.float32)
        output_f32_0 = vals_f32_0 * inv_rms_0 * weight_f32
        output_f32_1 = vals_f32_1 * inv_rms_1 * weight_f32
        output_f32_2 = vals_f32_2 * inv_rms_2 * weight_f32
        output_f32_3 = vals_f32_3 * inv_rms_3 * weight_f32
        output_0 = output_f32_0.to(vals_0.dtype)
        output_1 = output_f32_1.to(vals_1.dtype)
        output_2 = output_f32_2.to(vals_2.dtype)
        output_3 = output_f32_3.to(vals_3.dtype)
        tl.store(output_ptr + (row_idx_base + 0) * output_row_stride + col_idx, output_0, mask=mask)
        tl.store(output_ptr + (row_idx_base + 1) * output_row_stride + col_idx, output_1, mask=mask)
        tl.store(output_ptr + (row_idx_base + 2) * output_row_stride + col_idx, output_2, mask=mask)
        tl.store(output_ptr + (row_idx_base + 3) * output_row_stride + col_idx, output_3, mask=mask)
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
    BLOCK_SIZE: tl.constexpr = 512 if n_cols <= 2048 else 1024
    n_rows_per_pid: tl.constexpr = 4
    grid = ((n_rows + n_rows_per_pid - 1) // n_rows_per_pid,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)