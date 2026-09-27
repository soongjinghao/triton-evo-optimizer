import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel_packed(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    PACK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pack_id = tl.program_id(0).to(tl.int64)
    row_base = pack_id * PACK_SIZE
    row_offsets = tl.arange(0, PACK_SIZE)
    row_indices = row_base + row_offsets

    # Compute sum_sq for each row in the pack
    sum_sq = tl.zeros([PACK_SIZE], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask_col = col_idx < n_cols
        # Load a 2D block: PACK_SIZE rows x BLOCK_SIZE cols
        # Use 1D offsets: row_base * input_row_stride + row_offsets * input_row_stride + col_idx
        # Since input_row_stride is constant per row, we can compute base for each row
        row_bases = row_indices * input_row_stride
        # Expand to 2D: each row base + col_idx
        # We'll load row by row to avoid high-dim view
        for r in range(PACK_SIZE):
            row_ptr = input_ptr + row_bases[r]
            vals = tl.load(row_ptr + col_idx, mask=mask_col, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq = tl.where(tl.arange(0, PACK_SIZE) == r, sum_sq + tl.sum(tl.where(mask_col, sq_vals, 0.0)), sum_sq)

    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Write output for each row in the pack
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask_col = col_idx < n_cols
        weight_vals = tl.load(weight_ptr + col_idx, mask=mask_col, other=1.0)
        weight_f32 = weight_vals.to(tl.float32)
        for r in range(PACK_SIZE):
            row_ptr = input_ptr + row_indices[r] * input_row_stride
            out_row_ptr = output_ptr + row_indices[r] * output_row_stride
            vals = tl.load(row_ptr + col_idx, mask=mask_col, other=0.0)
            vals_f32 = vals.to(tl.float32)
            output_f32 = vals_f32 * inv_rms[r] * weight_f32
            output = output_f32.to(vals.dtype)
            tl.store(out_row_ptr + col_idx, output, mask=mask_col)

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
    PACK_SIZE = 4
    # Handle non-divisible n_rows by padding grid
    n_packs = (n_rows + PACK_SIZE - 1) // PACK_SIZE
    grid = (n_packs,)
    _rms_norm_kernel_packed[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        PACK_SIZE=PACK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)