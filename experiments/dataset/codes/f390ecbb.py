import logging
import torch
import triton
import triton.language as tl

@triton.jit
def post_reorder_triton_kernel_for_cutlass_moe(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    num_local_experts,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty
    group_idx = tl.program_id(0)
    token_base = group_idx * GROUP_SIZE
    vec = tl.arange(0, BLOCK_SIZE)
    token_offsets = token_base + tl.arange(0, GROUP_SIZE)[:, None]
    col_offsets = vec[None, :]
    mask = col_offsets < hidden_size
    token_mask = token_offsets < seq_len
    full_mask = token_mask & mask
    sum_vec = tl.zeros([GROUP_SIZE, BLOCK_SIZE], dtype=InDtype)
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + token_offsets * topk + idx, mask=token_mask, other=num_local_experts)
        is_valid = expert_id != num_local_experts
        dst_idx_int32 = tl.load(src2dst_ptr + token_offsets * topk + idx, mask=token_mask & is_valid, other=0)
        dst_idx = dst_idx_int32.to(tl.int64)
        weigh_scale = tl.load(topk_weights_ptr + token_offsets * topk + idx, mask=token_mask & is_valid, other=0.0).to(InDtype)
        load_ptr = down_output_ptr + dst_idx * hidden_size + col_offsets
        in_data = tl.load(load_ptr, mask=full_mask & is_valid[:, None], other=0.0)
        sum_vec += in_data * weigh_scale[:, None]
    store_ptr = output_ptr + token_offsets * hidden_size + col_offsets
    tl.store(store_ptr, sum_vec, mask=full_mask)

def post_reorder_triton_for_cutlass_moe(
    down_output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
):
    seq_len = src2dst.shape[0]
    topk = src2dst.shape[1]
    hidden_size = down_output.shape[-1]
    output = torch.empty((seq_len, hidden_size), dtype=down_output.dtype, device=down_output.device)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    GROUP_SIZE = 4
    grid = ((seq_len + GROUP_SIZE - 1) // GROUP_SIZE,)
    post_reorder_triton_kernel_for_cutlass_moe[grid](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        topk,
        num_local_experts,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
    )
    return output