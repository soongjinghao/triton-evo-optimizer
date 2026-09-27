import enum
import logging
from typing import Any, Iterable, Optional, Set, Tuple
import torch
from torch import nn
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_is_npu = True

@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    TOTAL_BATCH: tl.constexpr,
):
    i_full = tl.program_id(0)

    TOTAL_HEAD_BLOCKS: tl.constexpr = (NUM_HEADS_QK + BLOCK_HEADS - 1) // BLOCK_HEADS
    i_bs_block = i_full // TOTAL_HEAD_BLOCKS
    i_head_block = i_full % TOTAL_HEAD_BLOCKS

    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    for b_offset in range(BLOCK_B):
        b_idx = i_bs_block * BLOCK_B + b_offset
        if b_idx < TOTAL_BATCH:
            for h_offset in range(BLOCK_HEADS):
                i_qk = i_head_block * BLOCK_HEADS + h_offset
                if i_qk < NUM_HEADS_QK:
                    base_ptr = mixed_qkvz + b_idx * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T

                    q_end: tl.constexpr = HEAD_QK
                    k_end: tl.constexpr = q_end + HEAD_QK
                    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
                    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V

                    q_vals = tl.load(base_ptr + tl.arange(0, q_end))
                    k_vals = tl.load(base_ptr + tl.arange(q_end, k_end))
                    v_vals = tl.load(base_ptr + tl.arange(k_end, v_end))
                    z_vals = tl.load(base_ptr + tl.arange(v_end, z_end))

                    blk_q_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + i_qk * HEAD_QK
                    blk_k_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
                    blk_v_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK

                    tl.store(blk_q_st_ptr + tl.arange(0, HEAD_QK), q_vals)
                    tl.store(blk_k_st_ptr + tl.arange(0, HEAD_QK), k_vals)
                    tl.store(blk_v_st_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)

                    blk_z_st_ptr = z + b_idx * NUM_HEADS_V * HEAD_V + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
                    tl.store(blk_z_st_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)

                    ba_base_ptr = mixed_ba + b_idx * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
                    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
                    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK

                    b_vals = tl.load(ba_base_ptr + tl.arange(0, b_end))
                    a_vals = tl.load(ba_base_ptr + tl.arange(b_end, a_end))

                    blk_b_st_ptr = b + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK
                    blk_a_st_ptr = a + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK

                    tl.store(blk_b_st_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
                    tl.store(blk_a_st_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v

    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)

    BLOCK_B = 2
    if num_heads_qk % 2 == 0:
        BLOCK_HEADS = 2
    else:
        BLOCK_HEADS = 1

    TOTAL_BATCH = batch * seq_len
    grid = (
        triton.cdiv(TOTAL_BATCH, BLOCK_B) * triton.cdiv(num_heads_qk, BLOCK_HEADS),
    )

    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        BLOCK_B=BLOCK_B,
        BLOCK_HEADS=BLOCK_HEADS,
        TOTAL_BATCH=TOTAL_BATCH,
        num_warps=1,
        num_stages=3,
    )

    return mixed_qkv, z, b, a


ALL_DECODER_LAYER_TYPES = {
    "attention": None,
    "linear_attention": None,
}
EntryClass = None