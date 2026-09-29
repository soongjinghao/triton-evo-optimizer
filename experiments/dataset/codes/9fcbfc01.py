import torch
import triton
import triton.language as tl

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    # 将 query_start_len_ptr 作为循环不变量,避免每次迭代重新计算基地址
    base_ptr = query_start_len_ptr
    while left < right:
        # 部分展开:每次迭代比较两个 mid 点
        mid1 = (left + right) // 2
        mid2 = mid1 + 1
        val1 = tl.load(base_ptr + mid1)
        mid_val1 = val1 // BLOCK_Q + mid1 if use_q_block_mode else val1
        if mid_val1 <= target_idx:
            left = mid1 + 1
            # 检查 mid2 是否在范围内
            if mid2 < right:
                val2 = tl.load(base_ptr + mid2)
                mid_val2 = val2 // BLOCK_Q + mid2 if use_q_block_mode else val2
                if mid_val2 <= target_idx:
                    left = mid2 + 1
                else:
                    right = mid2
            else:
                # mid2 超出范围,直接结束
                break
        else:
            right = mid1
    return left - 1

@triton.jit
def reduce_segments(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    out_scale_inv,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    block_table_stride: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)
    # 将重复的 query_token_idx * (num_query_heads * NUM_SEGMENTS_PER_SEQ) 外提为循环不变量
    token_head_segm_base = query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
    segm_offset = (
        token_head_segm_base
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)
    # 将重复的 query_token_idx * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) 外提
    token_head_segm_head_base = query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
    segm_output_offset = (
        token_head_segm_head_base
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)