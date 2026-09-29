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
    group_id = tl.program_id(0)
    src_idx_base = group_id * GROUP_SIZE
    vec = tl.arange(0, BLOCK_SIZE)
    for i in range(GROUP_SIZE):
        src_idx_int32 = src_idx_base + i
        src_idx = src_idx_int32.to(tl.int64)
        src2dst_ptr_local = src2dst_ptr + src_idx * topk
        topk_ids_ptr_local = topk_ids_ptr + src_idx * topk
        topk_weights_ptr_local = topk_weights_ptr + src_idx * topk
        store_ptr = output_ptr + src_idx * hidden_size
        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + vec
            mask = offset < hidden_size
            sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
            for idx in range(topk):
                expert_id = tl.load(topk_ids_ptr_local + idx)
                if expert_id != num_local_experts:
                    dst_idx_int32 = tl.load(src2dst_ptr_local + idx)
                    dst_idx = dst_idx_int32.to(tl.int64)
                    weigh_scale = tl.load(topk_weights_ptr_local + idx).to(InDtype)
                    load_ptr = down_output_ptr + dst_idx * hidden_size
                    in_data = tl.load(load_ptr + offset, mask=mask)
                    sum_vec += in_data * weigh_scale
            tl.store(store_ptr + offset, sum_vec, mask=mask)

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
    grid = (triton.cdiv(seq_len, GROUP_SIZE),)
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