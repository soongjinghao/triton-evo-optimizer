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
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    q_end: tl.constexpr = HEAD_QK
    k_end: tl.constexpr = q_end + HEAD_QK
    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V
    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK

    if BLOCK_B == 2:
        qkvz_base = mixed_qkvz + i_qk * QKVZ_DIM_T
        qkv_q_base = mixed_qkv + i_qk * HEAD_QK
        qkv_k_base = mixed_qkv + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
        qkv_v_base = mixed_qkv + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
        z_base = z + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
        ba_base = mixed_ba + i_qk * BA_DIM_T
        out_b_base = b + i_qk * NUM_HEADS_V_PER_QK
        out_a_base = a + i_qk * NUM_HEADS_V_PER_QK

        qkvz_row_stride: tl.constexpr = NUM_HEADS_QK * QKVZ_DIM_T
        qkv_row_stride: tl.constexpr = NUM_HEADS_QK * QKV_DIM_T
        z_row_stride: tl.constexpr = NUM_HEADS_V * HEAD_V
        ba_row_stride: tl.constexpr = NUM_HEADS_QK * BA_DIM_T
        out_row_stride: tl.constexpr = NUM_HEADS_V

        b0 = i_bs_block * BLOCK_B
        b1 = b0 + 1

        if b0 < TOTAL_BATCH:
            qkvz_off0 = qkvz_base + b0 * qkvz_row_stride
            q0_vals = tl.load(qkvz_off0 + tl.arange(0, q_end))
            k0_vals = tl.load(qkvz_off0 + tl.arange(q_end, k_end))
            v0_vals = tl.load(qkvz_off0 + tl.arange(k_end, v_end))
            z0_vals = tl.load(qkvz_off0 + tl.arange(v_end, z_end))

            row_qkv_off0 = b0 * qkv_row_stride
            tl.store(qkv_q_base + row_qkv_off0 + tl.arange(0, HEAD_QK), q0_vals)
            tl.store(qkv_k_base + row_qkv_off0 + tl.arange(0, HEAD_QK), k0_vals)
            tl.store(qkv_v_base + row_qkv_off0 + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v0_vals)
            tl.store(z_base + b0 * z_row_stride + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z0_vals)

            ba_off0 = ba_base + b0 * ba_row_stride
            b0_vals = tl.load(ba_off0 + tl.arange(0, b_end))
            a0_vals = tl.load(ba_off0 + tl.arange(b_end, a_end))
            out_head_off0 = b0 * out_row_stride
            tl.store(out_b_base + out_head_off0 + tl.arange(0, NUM_HEADS_V_PER_QK), b0_vals)
            tl.store(out_a_base + out_head_off0 + tl.arange(0, NUM_HEADS_V_PER_QK), a0_vals)

        if b1 < TOTAL_BATCH:
            qkvz_off1 = qkvz_base + b1 * qkvz_row_stride
            q1_vals = tl.load(qkvz_off1 + tl.arange(0, q_end))
            k1_vals = tl.load(qkvz_off1 + tl.arange(q_end, k_end))
            v1_vals = tl.load(qkvz_off1 + tl.arange(k_end, v_end))
            z1_vals = tl.load(qkvz_off1 + tl.arange(v_end, z_end))

            row_qkv_off1 = b1 * qkv_row_stride
            tl.store(qkv_q_base + row_qkv_off1 + tl.arange(0, HEAD_QK), q1_vals)
            tl.store(qkv_k_base + row_qkv_off1 + tl.arange(0, HEAD_QK), k1_vals)
            tl.store(qkv_v_base + row_qkv_off1 + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v1_vals)
            tl.store(z_base + b1 * z_row_stride + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z1_vals)

            ba_off1 = ba_base + b1 * ba_row_stride
            b1_vals = tl.load(ba_off1 + tl.arange(0, b_end))
            a1_vals = tl.load(ba_off1 + tl.arange(b_end, a_end))
            out_head_off1 = b1 * out_row_stride
            tl.store(out_b_base + out_head_off1 + tl.arange(0, NUM_HEADS_V_PER_QK), b1_vals)
            tl.store(out_a_base + out_head_off1 + tl.arange(0, NUM_HEADS_V_PER_QK), a1_vals)
    else:
        for b_offset in range(BLOCK_B):
            b_idx = i_bs_block * BLOCK_B + b_offset
            if b_idx < TOTAL_BATCH:
                base_ptr = mixed_qkvz + b_idx * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T
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