import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def apply_expert_map(expert_id, expert_map):
    return tl.load(expert_map + expert_id)

@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < total_token_num

    for d_block in range(NUM_D_BLOCKS):
        off_d = tl.arange(0, BLOCK_D)
        for b_offset in range(BLOCK_B):
            b_token = b_start + b_offset
            if b_token < total_token_num:
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                # Unroll topk_num loop for compile-time constant topk_num <= 8
                if topk_num == 1:
                    expert_id = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    if HAS_EXPERT_MAP:
                        expert_id = apply_expert_map(expert_id, expert_map)
                    if expert_id >= 0:
                        source_token_index = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp.to(tl.float32) * acc_weight
                elif topk_num == 2:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                elif topk_num == 3:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                elif topk_num == 4:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    expert_id3 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 3)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                        expert_id3 = apply_expert_map(expert_id3, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                    if expert_id3 >= 0:
                        source_token_index3 = tl.load(input_index + b_token * input_index_stride0 + 3)
                        acc_weight3 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 3)
                        tmp3 = tl.load(input_tensor + source_token_index3 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp3.to(tl.float32) * acc_weight3
                elif topk_num == 5:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    expert_id3 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 3)
                    expert_id4 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 4)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                        expert_id3 = apply_expert_map(expert_id3, expert_map)
                        expert_id4 = apply_expert_map(expert_id4, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                    if expert_id3 >= 0:
                        source_token_index3 = tl.load(input_index + b_token * input_index_stride0 + 3)
                        acc_weight3 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 3)
                        tmp3 = tl.load(input_tensor + source_token_index3 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp3.to(tl.float32) * acc_weight3
                    if expert_id4 >= 0:
                        source_token_index4 = tl.load(input_index + b_token * input_index_stride0 + 4)
                        acc_weight4 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 4)
                        tmp4 = tl.load(input_tensor + source_token_index4 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp4.to(tl.float32) * acc_weight4
                elif topk_num == 6:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    expert_id3 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 3)
                    expert_id4 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 4)
                    expert_id5 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 5)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                        expert_id3 = apply_expert_map(expert_id3, expert_map)
                        expert_id4 = apply_expert_map(expert_id4, expert_map)
                        expert_id5 = apply_expert_map(expert_id5, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                    if expert_id3 >= 0:
                        source_token_index3 = tl.load(input_index + b_token * input_index_stride0 + 3)
                        acc_weight3 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 3)
                        tmp3 = tl.load(input_tensor + source_token_index3 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp3.to(tl.float32) * acc_weight3
                    if expert_id4 >= 0:
                        source_token_index4 = tl.load(input_index + b_token * input_index_stride0 + 4)
                        acc_weight4 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 4)
                        tmp4 = tl.load(input_tensor + source_token_index4 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp4.to(tl.float32) * acc_weight4
                    if expert_id5 >= 0:
                        source_token_index5 = tl.load(input_index + b_token * input_index_stride0 + 5)
                        acc_weight5 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 5)
                        tmp5 = tl.load(input_tensor + source_token_index5 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp5.to(tl.float32) * acc_weight5
                elif topk_num == 7:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    expert_id3 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 3)
                    expert_id4 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 4)
                    expert_id5 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 5)
                    expert_id6 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 6)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                        expert_id3 = apply_expert_map(expert_id3, expert_map)
                        expert_id4 = apply_expert_map(expert_id4, expert_map)
                        expert_id5 = apply_expert_map(expert_id5, expert_map)
                        expert_id6 = apply_expert_map(expert_id6, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                    if expert_id3 >= 0:
                        source_token_index3 = tl.load(input_index + b_token * input_index_stride0 + 3)
                        acc_weight3 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 3)
                        tmp3 = tl.load(input_tensor + source_token_index3 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp3.to(tl.float32) * acc_weight3
                    if expert_id4 >= 0:
                        source_token_index4 = tl.load(input_index + b_token * input_index_stride0 + 4)
                        acc_weight4 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 4)
                        tmp4 = tl.load(input_tensor + source_token_index4 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp4.to(tl.float32) * acc_weight4
                    if expert_id5 >= 0:
                        source_token_index5 = tl.load(input_index + b_token * input_index_stride0 + 5)
                        acc_weight5 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 5)
                        tmp5 = tl.load(input_tensor + source_token_index5 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp5.to(tl.float32) * acc_weight5
                    if expert_id6 >= 0:
                        source_token_index6 = tl.load(input_index + b_token * input_index_stride0 + 6)
                        acc_weight6 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 6)
                        tmp6 = tl.load(input_tensor + source_token_index6 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp6.to(tl.float32) * acc_weight6
                elif topk_num == 8:
                    expert_id0 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 0)
                    expert_id1 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 1)
                    expert_id2 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 2)
                    expert_id3 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 3)
                    expert_id4 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 4)
                    expert_id5 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 5)
                    expert_id6 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 6)
                    expert_id7 = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + 7)
                    if HAS_EXPERT_MAP:
                        expert_id0 = apply_expert_map(expert_id0, expert_map)
                        expert_id1 = apply_expert_map(expert_id1, expert_map)
                        expert_id2 = apply_expert_map(expert_id2, expert_map)
                        expert_id3 = apply_expert_map(expert_id3, expert_map)
                        expert_id4 = apply_expert_map(expert_id4, expert_map)
                        expert_id5 = apply_expert_map(expert_id5, expert_map)
                        expert_id6 = apply_expert_map(expert_id6, expert_map)
                        expert_id7 = apply_expert_map(expert_id7, expert_map)
                    if expert_id0 >= 0:
                        source_token_index0 = tl.load(input_index + b_token * input_index_stride0 + 0)
                        acc_weight0 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 0)
                        tmp0 = tl.load(input_tensor + source_token_index0 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp0.to(tl.float32) * acc_weight0
                    if expert_id1 >= 0:
                        source_token_index1 = tl.load(input_index + b_token * input_index_stride0 + 1)
                        acc_weight1 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 1)
                        tmp1 = tl.load(input_tensor + source_token_index1 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp1.to(tl.float32) * acc_weight1
                    if expert_id2 >= 0:
                        source_token_index2 = tl.load(input_index + b_token * input_index_stride0 + 2)
                        acc_weight2 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 2)
                        tmp2 = tl.load(input_tensor + source_token_index2 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp2.to(tl.float32) * acc_weight2
                    if expert_id3 >= 0:
                        source_token_index3 = tl.load(input_index + b_token * input_index_stride0 + 3)
                        acc_weight3 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 3)
                        tmp3 = tl.load(input_tensor + source_token_index3 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp3.to(tl.float32) * acc_weight3
                    if expert_id4 >= 0:
                        source_token_index4 = tl.load(input_index + b_token * input_index_stride0 + 4)
                        acc_weight4 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 4)
                        tmp4 = tl.load(input_tensor + source_token_index4 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp4.to(tl.float32) * acc_weight4
                    if expert_id5 >= 0:
                        source_token_index5 = tl.load(input_index + b_token * input_index_stride0 + 5)
                        acc_weight5 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 5)
                        tmp5 = tl.load(input_tensor + source_token_index5 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp5.to(tl.float32) * acc_weight5
                    if expert_id6 >= 0:
                        source_token_index6 = tl.load(input_index + b_token * input_index_stride0 + 6)
                        acc_weight6 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 6)
                        tmp6 = tl.load(input_tensor + source_token_index6 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp6.to(tl.float32) * acc_weight6
                    if expert_id7 >= 0:
                        source_token_index7 = tl.load(input_index + b_token * input_index_stride0 + 7)
                        acc_weight7 = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + 7)
                        tmp7 = tl.load(input_tensor + source_token_index7 * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                        accumulator += tmp7.to(tl.float32) * acc_weight7
                else:
                    for topk_index in range(0, topk_num):
                        expert_id = tl.load(recv_topk_ids + b_token * recv_topk_ids_stride0 + topk_index)
                        if HAS_EXPERT_MAP:
                            expert_id = apply_expert_map(expert_id, expert_map)
                        if expert_id >= 0:
                            source_token_index = tl.load(input_index + b_token * input_index_stride0 + topk_index)
                            acc_weight = tl.load(recv_topk_weight + b_token * recv_topk_weight_stride0 + topk_index)
                            tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + d_block * BLOCK_D + off_d)
                            accumulator += tmp.to(tl.float32) * acc_weight
                tl.store(
                    output_tensor + b_token * output_tensor_stride0 + d_block * BLOCK_D + off_d,
                    accumulator.to(output_tensor.dtype.element_ty),
                )

@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    assert input_tensor.device.type == 'npu', "input_tensor must be on NPU"
    assert recv_topk_ids.device.type == 'npu', "recv_topk_ids must be on NPU"
    assert recv_topk_weight.device.type == 'npu', "recv_topk_weight must be on NPU"
    assert input_index.device.type == 'npu', "input_index must be on NPU"
    assert output_tensor.device.type == 'npu', "output_tensor must be on NPU"
    if expert_map is not None:
        assert expert_map.device.type == 'npu', "expert_map must be on NPU"
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = min(hidden_size, 1024)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)
    BLOCK_B = 4
    grid = (triton.cdiv(num_tokens, BLOCK_B),)
    if grid[0] > 40:
        BLOCK_B = triton.cdiv(num_tokens, 40)
        grid = (40,)
    assert hidden_size % BLOCK_D == 0, f"hidden_size {hidden_size} must be divisible by BLOCK_D {BLOCK_D}"
    assert num_tokens > 0, "Cannot process empty token tensor"
    assert hidden_size > 0, "Cannot process empty hidden dimension"
    NUM_D_BLOCKS = triton.cdiv(hidden_size, BLOCK_D)
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return output_tensor