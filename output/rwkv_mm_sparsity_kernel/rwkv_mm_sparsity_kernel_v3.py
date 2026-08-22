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

    acc = tl.zeros((block_size,), dtype=tl.float32)

    num_blocks = tl.cdiv(k_size, blk_size)

    k_offset = tl.arange(0, blk_size)
    k_mask = k_offset < k_size
    current_k = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
    current_v = tl.load(
        v_ptr + k_offset[:, None] * v_cols + col_idx[None, :],
        mask=k_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    for i in range(1, num_blocks):
        next_k_offset = i * blk_size + tl.arange(0, blk_size)
        next_k_mask = next_k_offset < k_size
        next_k = tl.load(k_ptr + next_k_offset, mask=next_k_mask, other=0.0)
        next_v = tl.load(
            v_ptr + next_k_offset[:, None] * v_cols + col_idx[None, :],
            mask=next_k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        weighted_v = current_k[:, None] * current_v
        acc += tl.sum(weighted_v, axis=0)

        current_k = next_k
        current_v = next_v

    if num_blocks > 0:
        weighted_v = current_k[:, None] * current_v
        acc += tl.sum(weighted_v, axis=0)

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

    grid = (triton.cdiv(v_cols, block_size),)

    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        blk_size,
        k_size,
        block_size,
        num_warps=4,
        num_stages=2,
    )
    return output