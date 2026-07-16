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
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    if n_cols <= BLOCK_SIZE:
        # Single-pass path: entire row fits in one block
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        weight_f32 = weight.to(tl.float32)

        for r in range(ROWS_PER_BLOCK):
            row_idx = row_start + r
            row_valid = row_idx < n_rows
            load_mask = mask & row_valid
            input_row_ptr = input_ptr + row_idx * input_row_stride
            vals = tl.load(input_row_ptr + col_idx, mask=load_mask, other=0.0)
            vals_f32 = vals.to(tl.float32)

            sum_sq = tl.sum(vals_f32 * vals_f32)
            mean_sq = sum_sq / n_cols
            inv_rms = tl.rsqrt(mean_sq + eps)

            output_f32 = vals_f32 * inv_rms * weight_f32
            output = output_f32.to(vals.dtype)
            output_row_ptr = output_ptr + row_idx * output_row_stride
            tl.store(output_row_ptr + col_idx, output, mask=load_mask)
    else:
        # Two-pass path: n_cols > BLOCK_SIZE, need column loops
        # First pass: compute inv_rms for each row
        inv_rms = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
        for r in range(ROWS_PER_BLOCK):
            row_idx = row_start + r
            row_valid = row_idx < n_rows
            sum_sq = tl.zeros([1], dtype=tl.float32)
            for col_offset in range(0, n_cols, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                col_mask = col_idx < n_cols
                load_mask = col_mask & row_valid
                vals = tl.load(input_ptr + row_idx * input_row_stride + col_idx, mask=load_mask, other=0.0)
                vals_f32 = vals.to(tl.float32)
                sum_sq += tl.sum(vals_f32 * vals_f32)
            mean_sq = sum_sq / n_cols
            inv_rms_val = tl.rsqrt(mean_sq + eps)
            inv_rms = tl.where(tl.arange(0, ROWS_PER_BLOCK) == r, inv_rms_val, inv_rms)

        # Second pass: apply weight and inv_rms, sharing weight across rows
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < n_cols
            weight = tl.load(weight_ptr + col_idx, mask=col_mask, other=1.0)
            weight_f32 = weight.to(tl.float32)
            for r in range(ROWS_PER_BLOCK):
                row_idx = row_start + r
                row_valid = row_idx < n_rows
                load_mask = col_mask & row_valid
                vals = tl.load(input_ptr + row_idx * input_row_stride + col_idx, mask=load_mask, other=0.0)
                vals_f32 = vals.to(tl.float32)
                cur_inv_rms = tl.sum(tl.where(tl.arange(0, ROWS_PER_BLOCK) == r, inv_rms, 0.0))
                output_f32 = vals_f32 * cur_inv_rms * weight_f32
                output = output_f32.to(vals.dtype)
                tl.store(output_ptr + row_idx * output_row_stride + col_idx, output, mask=load_mask)


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

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    ROWS_PER_BLOCK = 4
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)

    num_warps = 4
    num_stages = 1

    _rms_norm_kernel[grid](
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
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.reshape(original_shape)