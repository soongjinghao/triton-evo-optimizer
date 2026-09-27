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

    QKVZ_ROW_STRIDE: tl.constexpr = NUM_HEADS_QK * QKVZ_DIM_T
    QKV_ROW_STRIDE: tl.constexpr = NUM_HEADS_QK * QKV_DIM_T
    BA_ROW_STRIDE: tl.constexpr = NUM_HEADS_QK * BA_DIM_T
    Z_ROW_STRIDE: tl.constexpr = NUM_HEADS_V * HEAD_V
    B_ROW_STRIDE: tl.constexpr = NUM_HEADS_V
    A_ROW_STRIDE: tl.constexpr = NUM_HEADS_V

    base_b_idx = i_bs_block * BLOCK_B

    qkvz_base = mixed_qkvz + base_b_idx * QKVZ_ROW_STRIDE + i_qk * QKVZ_DIM_T
    q_base = mixed_qkv + base_b_idx * QKV_ROW_STRIDE + i_qk * HEAD_QK
    k_base = mixed_qkv + base_b_idx * QKV_ROW_STRIDE + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
    v_base = mixed_qkv + base_b_idx * QKV_ROW_STRIDE + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    z_base = z + base_b_idx * Z_ROW_STRIDE + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    ba_base = mixed_ba + base_b_idx * BA_ROW_STRIDE + i_qk * BA_DIM_T
    b_base = b + base_b_idx * B_ROW_STRIDE + i_qk * NUM_HEADS_V_PER_QK
    a_base = a + base_b_idx * A_ROW_STRIDE + i_qk * NUM_HEADS_V_PER_QK

    q_end: tl.constexpr = HEAD_QK
    k_end: tl.constexpr = q_end + HEAD_QK
    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V
    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK

    FULL_BS_BLOCKS: tl.constexpr = TOTAL_BATCH // BLOCK_B

    if i_bs_block < FULL_BS_BLOCKS:
        qkvz_row_ptr = qkvz_base
        q_row_ptr = q_base
        k_row_ptr = k_base
        v_row_ptr = v_base
        z_row_ptr = z_base
        ba_row_ptr = ba_base
        b_row_ptr = b_base
        a_row_ptr = a_base

        for b_offset in range(BLOCK_B):
            q_vals = tl.load(qkvz_row_ptr + tl.arange(0, q_end))
            k_vals = tl.load(qkvz_row_ptr + tl.arange(q_end, k_end))
            v_vals = tl.load(qkvz_row_ptr + tl.arange(k_end, v_end))
            z_vals = tl.load(qkvz_row_ptr + tl.arange(v_end, z_end))

            tl.store(q_row_ptr + tl.arange(0, HEAD_QK), q_vals)
            tl.store(k_row_ptr + tl.arange(0, HEAD_QK), k_vals)
            tl.store(v_row_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)
            tl.store(z_row_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)

            b_vals = tl.load(ba_row_ptr + tl.arange(0, b_end))
            a_vals = tl.load(ba_row_ptr + tl.arange(b_end, a_end))
            tl.store(b_row_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
            tl.store(a_row_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)

            qkvz_row_ptr = qkvz_row_ptr + QKVZ_ROW_STRIDE
            q_row_ptr = q_row_ptr + QKV_ROW_STRIDE
            k_row_ptr = k_row_ptr + QKV_ROW_STRIDE
            v_row_ptr = v_row_ptr + QKV_ROW_STRIDE
            z_row_ptr = z_row_ptr + Z_ROW_STRIDE
            ba_row_ptr = ba_row_ptr + BA_ROW_STRIDE
            b_row_ptr = b_row_ptr + B_ROW_STRIDE
            a_row_ptr = a_row_ptr + A_ROW_STRIDE
    else:
        qkvz_row_ptr = qkvz_base
        q_row_ptr = q_base
        k_row_ptr = k_base
        v_row_ptr = v_base
        z_row_ptr = z_base
        ba_row_ptr = ba_base
        b_row_ptr = b_base
        a_row_ptr = a_base

        for b_offset in range(BLOCK_B):
            b_idx = base_b_idx + b_offset
            if b_idx < TOTAL_BATCH:
                q_vals = tl.load(qkvz_row_ptr + tl.arange(0, q_end))
                k_vals = tl.load(qkvz_row_ptr + tl.arange(q_end, k_end))
                v_vals = tl.load(qkvz_row_ptr + tl.arange(k_end, v_end))
                z_vals = tl.load(qkvz_row_ptr + tl.arange(v_end, z_end))

                tl.store(q_row_ptr + tl.arange(0, HEAD_QK), q_vals)
                tl.store(k_row_ptr + tl.arange(0, HEAD_QK), k_vals)
                tl.store(v_row_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)
                tl.store(z_row_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)

                b_vals = tl.load(ba_row_ptr + tl.arange(0, b_end))
                a_vals = tl.load(ba_row_ptr + tl.arange(b_end, a_end))
                tl.store(b_row_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
                tl.store(a_row_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)

            qkvz_row_ptr = qkvz_row_ptr + QKVZ_ROW_STRIDE
            q_row_ptr = q_row_ptr + QKV_ROW_STRIDE
            k_row_ptr = k_row_ptr + QKV_ROW_STRIDE
            v_row_ptr = v_row_ptr + QKV_ROW_STRIDE
            z_row_ptr = z_row_ptr + Z_ROW_STRIDE
            ba_row_ptr = ba_row_ptr + BA_ROW_STRIDE
            b_row_ptr = b_row_ptr + B_ROW_STRIDE
            a_row_ptr = a_row_ptr + A_ROW_STRIDE
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