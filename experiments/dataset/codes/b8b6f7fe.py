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
    num_col_blocks = tl.cdiv(v_cols, block_size)
    col_block_id = pid % num_col_blocks
    k_block_id = pid // num_col_blocks
    col_idx = col_block_id * block_size + tl.arange(0, block_size)
    col_mask = col_idx < v_cols
    k_start = k_block_id * blk_size
    k_offset = k_start + tl.arange(0, blk_size)
    k_mask = k_offset < k_size
    k = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
    k_nonzero_mask = k != 0
    v_block = tl.load(v_ptr + k_offset[:, None] * v_cols + col_idx[None, :],
                      mask=k_mask[:, None] & col_mask[None, :], other=0.0)
    k_broadcast = tl.broadcast_to(k[:, None], (blk_size, block_size))
    weighted_v = tl.where(k_nonzero_mask[:, None] & col_mask[None, :],
                          k_broadcast.to(tl.float32) * v_block.to(tl.float32), 0.0)
    acc = tl.sum(weighted_v, axis=0)
    out_ptr = output_ptr + col_idx
    tl.store(out_ptr, acc, mask=col_mask)
def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)
    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)
    blk_size = 128
    block_size = 64
    k_size = k.size(0)
    num_col_blocks = triton.cdiv(v_cols, block_size)
    num_k_blocks = triton.cdiv(k_size, blk_size)
    grid = (num_col_blocks * num_k_blocks,)
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