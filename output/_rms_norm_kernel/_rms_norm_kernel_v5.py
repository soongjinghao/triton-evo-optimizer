import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel_full_row(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    col_offs = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offs < n_cols
    weight = tl.load(tl.multiple_of(weight_ptr + col_offs, 16), mask=col_mask, other=1.0)
    weight_f32 = weight.to(tl.float32)
    for r in tl.static_range(ROWS_PER_BLOCK):
        row_idx = row_start + r
        if row_idx < n_rows:
            row_start_ptr = input_ptr + row_idx * input_row_stride
            row_ptr = tl.multiple_of(row_start_ptr + col_offs, 16)
            vals = tl.load(row_ptr, mask=col_mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sum_sq = tl.sum(vals_f32 * vals_f32)
            inv_rms = tl.math.rsqrt(sum_sq / n_cols + eps)
            out_f32 = vals_f32 * inv_rms * weight_f32
            out = out_f32.to(vals.dtype)
            out_start_ptr = output_ptr + row_idx * output_row_stride
            out_ptr = tl.multiple_of(out_start_ptr + col_offs, 16)
            tl.store(out_ptr, out, mask=col_mask)

@triton.jit
def _rms_norm_kernel_masked(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    for r in tl.static_range(ROWS_PER_BLOCK):
        row_idx = row_start + r
        if row_idx < n_rows:
            row_start_ptr = input_ptr + row_idx * input_row_stride
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            sum_sq = tl.zeros([1], dtype=tl.float32)
            for col_offset in range(0, n_cols, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                mask = col_idx < n_cols
                vals = tl.load(tl.multiple_of(row_start_ptr + col_idx, 16), mask=mask, other=0.0)
                vals_f32 = vals.to(tl.float32)
                sum_sq += tl.sum(vals_f32 * vals_f32)
            inv_rms = tl.math.rsqrt(sum_sq / n_cols + eps)
            for col_offset in range(0, n_cols, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                mask = col_idx < n_cols
                vals = tl.load(tl.multiple_of(row_start_ptr + col_idx, 16), mask=mask, other=0.0)
                weight = tl.load(tl.multiple_of(weight_ptr + col_idx, 16), mask=mask, other=1.0)
                vals_f32 = vals.to(tl.float32)
                weight_f32 = weight.to(tl.float32)
                out_f32 = vals_f32 * inv_rms * weight_f32
                out = out_f32.to(vals.dtype)
                tl.store(tl.multiple_of(output_row_start_ptr + col_idx, 16), out, mask=mask)

@triton.jit
def _rms_norm_kernel_unmasked(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    for r in tl.static_range(ROWS_PER_BLOCK):
        row_idx = row_start + r
        if row_idx < n_rows:
            row_start_ptr = input_ptr + row_idx * input_row_stride
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            sum_sq = tl.zeros([1], dtype=tl.float32)
            for col_offset in range(0, n_cols, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                vals = tl.load(tl.multiple_of(row_start_ptr + col_idx, 16))
                vals_f32 = vals.to(tl.float32)
                sum_sq += tl.sum(vals_f32 * vals_f32)
            inv_rms = tl.math.rsqrt(sum_sq / n_cols + eps)
            for col_offset in range(0, n_cols, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                vals = tl.load(tl.multiple_of(row_start_ptr + col_idx, 16))
                weight = tl.load(tl.multiple_of(weight_ptr + col_idx, 16))
                vals_f32 = vals.to(tl.float32)
                weight_f32 = weight.to(tl.float32)
                out_f32 = vals_f32 * inv_rms * weight_f32
                out = out_f32.to(vals.dtype)
                tl.store(tl.multiple_of(output_row_start_ptr + col_idx, 16), out)

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
    if n_cols <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        ROWS_PER_BLOCK = 4
        grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
        _rms_norm_kernel_full_row[grid](
            input_2d,
            weight,
            output,
            input_2d.stride(0),
            output.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        )
    else:
        BLOCK_SIZE = 1024
        ROWS_PER_BLOCK = 4
        grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
        if n_cols % BLOCK_SIZE == 0:
            _rms_norm_kernel_unmasked[grid](
                input_2d,
                weight,
                output,
                input_2d.stride(0),
                output.stride(0),
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                ROWS_PER_BLOCK=ROWS_PER_BLOCK,
            )
        else:
            _rms_norm_kernel_masked[grid](
                input_2d,
                weight,
                output,
                input_2d.stride(0),
                output.stride(0),
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                ROWS_PER_BLOCK=ROWS_PER_BLOCK,
            )
    return output.reshape(original_shape)