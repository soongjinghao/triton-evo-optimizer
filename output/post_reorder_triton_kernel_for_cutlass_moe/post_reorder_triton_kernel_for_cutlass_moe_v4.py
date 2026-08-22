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
):
    InDtype = down_output_ptr.dtype.element_ty
    src_idx_int32 = tl.program_id(0)
    src_idx = src_idx_int32.to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk
    store_ptr = output_ptr + src_idx * hidden_size

    vec = tl.arange(0, BLOCK_SIZE)

    if hidden_size == BLOCK_SIZE:
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx_int32 = tl.load(src2dst_ptr + idx)
                dst_idx = dst_idx_int32.to(tl.int64)
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + vec)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + vec, sum_vec)
    else:
        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + vec
            mask = offset < hidden_size
            sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
            for idx in range(topk):
                expert_id = tl.load(topk_ids_ptr + idx)
                if expert_id != num_local_experts:
                    dst_idx_int32 = tl.load(src2dst_ptr + idx)
                    dst_idx = dst_idx_int32.to(tl.int64)
                    weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
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
    grid = (seq_len,)
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
    )
    return output