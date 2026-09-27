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
    GROUPS_PER_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    group_size = BLOCK_SIZE
    packed_size = GROUPS_PER_BLOCK * group_size
    
    sum_sq = tl.zeros([GROUPS_PER_BLOCK], dtype=tl.float32)
    
    for col_offset in range(0, n_cols, packed_size):
        col_idx = col_offset + tl.arange(0, packed_size)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        
        sq_2d = tl.view(sq_vals, (GROUPS_PER_BLOCK, group_size))
        mask_2d = tl.view(mask, (GROUPS_PER_BLOCK, group_size))
        
        sum_sq += tl.sum(tl.where(mask_2d, sq_2d, 0.0), axis=1)
    
    n_groups = (n_cols + group_size - 1) // group_size
    n_full_groups = n_groups // GROUPS_PER_BLOCK * GROUPS_PER_BLOCK
    tail_groups = n_groups - n_full_groups
    
    mean_sq = tl.where(
        tl.arange(0, GROUPS_PER_BLOCK) < n_full_groups // GROUPS_PER_BLOCK * GROUPS_PER_BLOCK // GROUPS_PER_BLOCK,
        sum_sq / group_size,
0
    )
    
    for g in range(GROUPS_PER_BLOCK):
        if g < n_full_groups // GROUPS_PER_BLOCK * GROUPS_PER_BLOCK // GROUPS_PER_BLOCK:
            g_start = g * group_size
            g_end = g_start + group_size
            if g_end <= n_cols:
                g_mask = (tl.arange(0, group_size) + g_start) < n_cols
                vals = tl.load(row_start_ptr + g_start + tl.arange(0, group_size), mask=g_mask, other=0.0)
                weight = tl.load(weight_ptr + g_start + tl.arange(0, group_size), mask=g_mask, other=1.0)
                vals_f32 = vals.to(tl.float32)
                weight_f32 = weight.to(tl.float32)
                rms = tl.sqrt(mean_sq[g] + eps)
                inv_rms = 1.0 / rms
                output_f32 = vals_f32 * inv_rms * weight_f32
                output = output_f32.to(vals.dtype)
                tl.store(output_row_start_ptr + g_start + tl.arange(0, group_size), output, mask=g_mask)
    
    if tail_groups > 0:
        tail_start = n_full_groups * group_size
        for col_offset in range(tail_start, n_cols, group_size):
            col_idx = col_offset + tl.arange(0, group_size)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            
            tail_sum_sq = tl.sum(tl.where(mask, vals_f32 * vals_f32, 0.0))
            tail_mean_sq = tail_sum_sq / group_size
            rms = tl.sqrt(tail_mean_sq + eps)
            inv_rms = 1.0 / rms
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
    GROUPS_PER_BLOCK = 4
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
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
    )
    return output.reshape(original_shape)

def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return rms_norm(input, weight, eps=eps)