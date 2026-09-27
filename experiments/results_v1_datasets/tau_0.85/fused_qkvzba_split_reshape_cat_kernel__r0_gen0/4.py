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
    TOTAL_BATCH: tl.constexpr,
):
    i_full = tl.program_id(0)
    TOTAL_BS_BLOCKS: tl.constexpr = (TOTAL_BATCH + BLOCK_B - 1) // BLOCK_B
    i_bs_block = i_full // NUM_HEADS_QK
    i_qk = i_full % NUM_HEADS_QK

    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + (NUM_HEADS_V // NUM_HEADS_QK) * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + (NUM_HEADS_V // NUM_HEADS_QK) * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    V_DIM: tl.constexpr = NUM_HEADS_V_PER_QK * HEAD_V
    BLOCK_H: tl.constexpr = 4

    if i_qk % BLOCK_H == 0:
        head_off = tl.arange(0, BLOCK_H)[:, None]
        head_mask = (i_qk + head_off) < NUM_HEADS_QK

        q_out_col = tl.arange(0, HEAD_QK)[None, :]
        k_in_col = q_out_col + HEAD_QK
        v_in_col = tl.arange(0, V_DIM)[None, :] + HEAD_QK * 2
        z_in_col = v_in_col + V_DIM
        v_out_col = tl.arange(0, V_DIM)[None, :]
        b_col = tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]
        a_in_col = b_col + NUM_HEADS_V_PER_QK

        for b_offset in range(BLOCK_B):
            b_idx = i_bs_block * BLOCK_B + b_offset
            if b_idx < TOTAL_BATCH:
                base_ptr = mixed_qkvz + b_idx * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T
                q_vals = tl.load(base_ptr + head_off * QKVZ_DIM_T + q_out_col, mask=head_mask)
                k_vals = tl.load(base_ptr + head_off * QKVZ_DIM_T + k_in_col, mask=head_mask)
                v_vals = tl.load(base_ptr + head_off * QKVZ_DIM_T + v_in_col, mask=head_mask)
                z_vals = tl.load(base_ptr + head_off * QKVZ_DIM_T + z_in_col, mask=head_mask)

                blk_q_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + i_qk * HEAD_QK
                blk_k_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
                blk_v_st_ptr = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * V_DIM
                tl.store(blk_q_st_ptr + head_off * HEAD_QK + q_out_col, q_vals, mask=head_mask)
                tl.store(blk_k_st_ptr + head_off * HEAD_QK + q_out_col, k_vals, mask=head_mask)
                tl.store(blk_v_st_ptr + head_off * V_DIM + v_out_col, v_vals, mask=head_mask)

                blk_z_st_ptr = z + b_idx * NUM_HEADS_V * HEAD_V + i_qk * V_DIM
                tl.store(blk_z_st_ptr + head_off * V_DIM + v_out_col, z_vals, mask=head_mask)

                ba_base_ptr = mixed_ba + b_idx * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
                b_vals = tl.load(ba_base_ptr + head_off * BA_DIM_T + b_col, mask=head_mask)
                a_vals = tl.load(ba_base_ptr + head_off * BA_DIM_T + a_in_col, mask=head_mask)

                blk_b_st_ptr = b + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK
                blk_a_st_ptr = a + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK
                tl.store(blk_b_st_ptr + head_off * NUM_HEADS_V_PER_QK + b_col, b_vals, mask=head_mask)
                tl.store(blk_a_st_ptr + head_off * NUM_HEADS_V_PER_QK + b_col, a_vals, mask=head_mask)
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
    TOTAL_BATCH = batch * seq_len
    grid = (triton.cdiv(TOTAL_BATCH, BLOCK_B) * num_heads_qk,)
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