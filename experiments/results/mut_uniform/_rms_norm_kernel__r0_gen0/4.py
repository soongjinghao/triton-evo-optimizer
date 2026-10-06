import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    col_idx = tl.program_id(0).to(tl.int64)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        row_idx = row_offset + tl.arange(0, BLOCK_SIZE)
        mask = row_idx < n_rows
        vals = tl.load(input_ptr + col_idx * n_rows + row_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_rows
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        row_idx = row_offset + tl.arange(0, BLOCK_SIZE)
        mask = row_idx < n_rows
        vals = tl.load(input_ptr + col_idx * n_rows + row_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=None, other=1.0)
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_ptr + col_idx * n_rows + row_idx, output, mask=mask)

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
    n_rows, n_cols = input_2d.shape
    input_t = input_2d.t().contiguous()
    weight = weight.contiguous()
    output_t = torch.empty_like(input_t)
    BLOCK_SIZE = 256
    grid = (n_cols,)
    _rms_norm_kernel[grid](
        input_t,
        weight,
        output_t,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output = output_t.t().contiguous()
    return output.reshape(original_shape)