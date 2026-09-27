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
    K: tl.constexpr,
):
    row_pack_idx = tl.program_id(0).to(tl.int64)
    base_row_idx = row_pack_idx * K
    pack_size = K * BLOCK_SIZE
    sum_sq = tl.zeros([K], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        for k in range(K):
            row_idx = base_row_idx + k
            row_start_ptr = input_ptr + row_idx * input_row_stride
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq_k = tl.sum(tl.where(mask, sq_vals, 0.0))
            sum_sq = tl.where(tl.arange(0, K) == k, sum_sq + sum_sq_k, sum_sq)
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        weight_f32 = weight.to(tl.float32)
        for k in range(K):
            row_idx = base_row_idx + k
            row_start_ptr = input_ptr + row_idx * input_row_stride
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            output_f32 = vals_f32 * inv_rms[k] * weight_f32
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
    K = 4
    n_packs = (n_rows + K - 1) // K
    grid = (n_packs,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        K=K,
    )
    return output.reshape(original_shape)