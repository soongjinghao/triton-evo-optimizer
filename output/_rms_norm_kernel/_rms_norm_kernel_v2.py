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
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    n_rows,
):
    row_start = tl.program_id(0).to(tl.int64) * ROWS_PER_BLOCK
    row_idx = row_start + tl.arange(0, ROWS_PER_BLOCK)[:, None]
    row_mask = row_idx < n_rows

    sum_sq = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < n_cols
        mask = row_mask & col_mask[None, :]
        input_ptrs = input_ptr + row_idx * input_row_stride + col_idx[None, :]
        vals_orig = tl.load(input_ptrs, mask=mask, other=0.0)
        vals_f32 = vals_orig.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq = sum_sq + tl.sum(sq_vals, axis=1)

    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < n_cols
        mask = row_mask & col_mask[None, :]
        weight = tl.load(weight_ptr + col_idx, mask=col_mask, other=1.0).to(tl.float32)
        input_ptrs = input_ptr + row_idx * input_row_stride + col_idx[None, :]
        vals_orig = tl.load(input_ptrs, mask=mask, other=0.0)
        vals_f32 = vals_orig.to(tl.float32)
        output_f32 = vals_f32 * inv_rms[:, None] * weight[None, :]
        output_orig = output_f32.to(vals_orig.dtype)
        output_ptrs = output_ptr + row_idx * output_row_stride + col_idx[None, :]
        tl.store(output_ptrs, output_orig, mask=mask)

def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert weight.dim() == 1
    assert input.shape[-1] == weight.shape[0]
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    ROWS_PER_BLOCK = 4
    BLOCK_SIZE = min(n_cols, 512)
    BLOCK_SIZE = max(BLOCK_SIZE, 1)
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        n_rows=n_rows,
        num_stages=1,
        num_warps=2,
    )
    return output.reshape(original_shape)

def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return rms_norm(input, weight, eps=eps)