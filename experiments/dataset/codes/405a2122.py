import torch
import triton
import triton.language as tl

@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    blk_size: tl.constexpr,
    k_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    col_idx = pid * block_size + tl.arange(0, block_size)
    col_mask = col_idx < v_cols
    
    # Precompute v_cols stride for reuse
    v_stride = v_cols
    
    acc = tl.zeros((block_size,), dtype=tl.float32)
    
    # Use tl.cdiv for loop bound
    num_blks = tl.cdiv(k_size, blk_size)
    
    for i in range(0, num_blks):
        k_offset = i * blk_size + tl.arange(0, blk_size)
        k_mask = k_offset < k_size
        k = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
        
        # Compute non-zero mask once
        k_nonzero_mask = k != 0
        
        # Load v in a more efficient manner: load contiguous blocks per row
        # Since v is 2D with shape (k_size, v_cols), we load each row's block
        v_vals = tl.zeros((blk_size, block_size), dtype=tl.float32)
        
        for j in range(0, blk_size):
            k_off = k_offset[j]
            k_mask_j = k_off < k_size
            k_nonzero_j = k_nonzero_mask[j]
            # Only load if k is non-zero and within bounds
            if k_mask_j and k_nonzero_j:
                v_row_ptr = v_ptr + k_off * v_stride + col_idx
                v_row = tl.load(v_row_ptr, mask=col_mask, other=0.0)
                v_vals = tl.where(tl.arange(0, blk_size)[:, None] == j, v_row[None, :], v_vals)
        
        # Compute contribution: k_j * v_j for each column
        k_float = k.to(tl.float32)
        v_float = v_vals.to(tl.float32)
        
        # Accumulate: sum over k dimension for each column
        # Use tl.sum with axis=0 to sum over rows
        acc += tl.sum(k_float[:, None] * v_float, axis=0)
    
    out_ptr = output_ptr + col_idx
    tl.store(out_ptr, acc, mask=col_mask)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)
    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)
    blk_size = triton.next_power_of_2(512)
    block_size = triton.next_power_of_2(16)
    k_size = triton.next_power_of_2(k.size(0))
    grid = (triton.cdiv(v_cols, block_size),)
    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        blk_size,
        k_size,
        block_size,
    )
    return output