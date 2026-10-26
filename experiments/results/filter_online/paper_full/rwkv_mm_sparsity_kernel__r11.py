import torch
import triton
import triton.language as tl

@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_T_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    blk_size: tl.constexpr,
    k_size: tl.constexpr,
    block_size: tl.constexpr,
    num_blks: tl.constexpr,
):
    pid = tl.program_id(0)
    col_idx = pid * block_size + tl.arange(0, block_size)
    col_mask = col_idx < v_cols
    acc = tl.zeros((block_size,), dtype=tl.float32)
    for i in range(0, num_blks, 2):
        k_offset_0 = i * blk_size + tl.arange(0, blk_size)
        k_mask_0 = k_offset_0 < k_size
        k_0 = tl.load(k_ptr + k_offset_0, mask=k_mask_0, other=0.0)
        k_nonzero_mask_0 = k_0 != 0
        v_T_ptr_block_0 = v_T_ptr + col_idx[:, None] * k_size + k_offset_0[None, :]
        v_T_0 = tl.load(
            v_T_ptr_block_0,
            mask=col_mask[:, None] & k_mask_0[None, :] & k_nonzero_mask_0[None, :],
            other=0.0,
        )
        acc += tl.sum(k_0[None, :].to(tl.float32) * v_T_0.to(tl.float32), axis=1)
        i_next = i + 1
        if i_next < num_blks:
            k_offset_1 = i_next * blk_size + tl.arange(0, blk_size)
            k_mask_1 = k_offset_1 < k_size
            k_1 = tl.load(k_ptr + k_offset_1, mask=k_mask_1, other=0.0)
            k_nonzero_mask_1 = k_1 != 0
            v_T_ptr_block_1 = v_T_ptr + col_idx[:, None] * k_size + k_offset_1[None, :]
            v_T_1 = tl.load(
                v_T_ptr_block_1,
                mask=col_mask[:, None] & k_mask_1[None, :] & k_nonzero_mask_1[None, :],
                other=0.0,
            )
            acc += tl.sum(k_1[None, :].to(tl.float32) * v_T_1.to(tl.float32), axis=1)
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
    num_blks = triton.cdiv(k_size, blk_size)
    v_T = v.T.contiguous()
    grid = (triton.cdiv(v_cols, block_size),)
    rwkv_mm_sparsity_kernel[grid](
        k,
        v_T,
        output,
        v_cols,
        blk_size,
        k_size,
        block_size,
        num_blks,
    )
    return output