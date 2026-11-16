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
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
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
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < act_num_segments
    dim_mask = tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    if act_num_segments == NUM_SEGMENTS_PER_SEQ:
        segm_max = tl.load(segm_max_ptr + segm_offset)
        segm_expsum = tl.load(segm_expsum_ptr + segm_offset)
    else:
        segm_max = tl.load(
            segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf")
        )
        segm_expsum = tl.load(
            segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0
        )
    overall_max = tl.max(segm_max)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    if HEAD_SIZE_PADDED == HEAD_SIZE:
        if act_num_segments == NUM_SEGMENTS_PER_SEQ:
            segm_output = tl.load(segm_output_ptr + segm_output_offset)
        else:
            segm_output = tl.load(
                segm_output_ptr + segm_output_offset,
                mask=segm_mask[:, None],
                other=0.0,
            )
    else:
        if act_num_segments == NUM_SEGMENTS_PER_SEQ:
            segm_output = tl.load(
                segm_output_ptr + segm_output_offset,
                mask=dim_mask[None, :],
                other=0.0,
            )
        else:
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
    if HEAD_SIZE_PADDED == HEAD_SIZE:
        tl.store(output_ptr + output_offset, acc)
    else:
        tl.store(output_ptr + output_offset, acc, mask=dim_mask)