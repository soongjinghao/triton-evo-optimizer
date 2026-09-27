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
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_idx = tl.program_id(0).to(tl.int64)
    row_idx = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx[:, None] < n_rows

    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < n_cols
        mask = row_mask & col_mask[None, :]
        row_offsets = row_idx[:, None] * input_row_stride
        col_offsets = col_idx[None, :]
        load_ptrs = input_ptr + row_offsets + col_offsets
        vals = tl.load(load_ptrs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0), axis=1)

    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < n_cols
        mask = row_mask & col_mask[None, :]
        row_offsets = row_idx[:, None] * input_row_stride
        col_offsets = col_idx[None, :]
        load_ptrs = input_ptr + row_offsets + col_offsets
        vals = tl.load(load_ptrs, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=col_mask, other=1.0)
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms[:, None] * weight_f32
        output = output_f32.to(vals.dtype)
        store_ptrs = output_ptr + row_offsets + col_offsets
        tl.store(store_ptrs, output, mask=mask)

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
    BLOCK_M = 4
    BLOCK_SIZE = 1024
    grid = ((n_rows + BLOCK_M - 1) // BLOCK_M,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)