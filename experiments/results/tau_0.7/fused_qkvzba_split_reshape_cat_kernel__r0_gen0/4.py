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

    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V
    V_HEAD_DIM: tl.constexpr = NUM_HEADS_V_PER_QK * HEAD_V

    if BLOCK_B == 2:
        for b_offset in range(BLOCK_B):
            b_idx = i_bs_block * BLOCK_B + b_offset
            if b_idx < TOTAL_BATCH:
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
    else:
        offs_b = tl.arange(0, BLOCK_B)
        b_idx = i_bs_block * BLOCK_B + offs_b
        mask_b = b_idx < TOTAL_BATCH

        offs_qk = tl.arange(0, HEAD_QK)
        offs_v = tl.arange(0, V_HEAD_DIM)
        offs_ba = tl.arange(0, NUM_HEADS_V_PER_QK)

        q_ptrs = mixed_qkvz + b_idx[:, None] * (NUM_HEADS_QK * QKVZ_DIM_T) + i_qk * QKVZ_DIM_T + offs_qk[None, :]
        q_mask = mask_b[:, None]
        q_vals = tl.load(q_ptrs, mask=q_mask)

        k_ptrs = mixed_qkvz + b_idx[:, None] * (NUM_HEADS_QK * QKVZ_DIM_T) + i_qk * QKVZ_DIM_T + HEAD_QK + offs_qk[None, :]
        k_vals = tl.load(k_ptrs, mask=q_mask)

        v_ptrs = mixed_qkvz + b_idx[:, None] * (NUM_HEADS_QK * QKVZ_DIM_T) + i_qk * QKVZ_DIM_T + HEAD_QK * 2 + offs_v[None, :]
        v_vals = tl.load(v_ptrs, mask=q_mask)

        z_ptrs = mixed_qkvz + b_idx[:, None] * (NUM_HEADS_QK * QKVZ_DIM_T) + i_qk * QKVZ_DIM_T + HEAD_QK * 2 + V_HEAD_DIM + offs_v[None, :]
        z_vals = tl.load(z_ptrs, mask=q_mask)

        q_out_ptrs = mixed_qkv + b_idx[:, None] * (NUM_HEADS_QK * QKV_DIM_T) + i_qk * HEAD_QK + offs_qk[None, :]
        tl.store(q_out_ptrs, q_vals, mask=q_mask)

        k_out_ptrs = mixed_qkv + b_idx[:, None] * (NUM_HEADS_QK * QKV_DIM_T) + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK + offs_qk[None, :]
        tl.store(k_out_ptrs, k_vals, mask=q_mask)

        v_out_ptrs = mixed_qkv + b_idx[:, None] * (NUM_HEADS_QK * QKV_DIM_T) + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * V_HEAD_DIM + offs_v[None, :]
        tl.store(v_out_ptrs, v_vals, mask=q_mask)

        z_out_ptrs = z + b_idx[:, None] * (NUM_HEADS_V * HEAD_V) + i_qk * V_HEAD_DIM + offs_v[None, :]
        tl.store(z_out_ptrs, z_vals, mask=q_mask)

        ba_ptrs = mixed_ba + b_idx[:, None] * (NUM_HEADS_QK * BA_DIM_T) + i_qk * BA_DIM_T + offs_ba[None, :]
        b_vals = tl.load(ba_ptrs, mask=q_mask)

        a_ptrs = mixed_ba + b_idx[:, None] * (NUM_HEADS_QK * BA_DIM_T) + i_qk * BA_DIM_T + NUM_HEADS_V_PER_QK + offs_ba[None, :]
        a_vals = tl.load(a_ptrs, mask=q_mask)

        b_out_ptrs = b + b_idx[:, None] * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK + offs_ba[None, :]
        a_out_ptrs = a + b_idx[:, None] * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK + offs_ba[None, :]
        tl.store(b_out_ptrs, b_vals, mask=q_mask)
        tl.store(a_out_ptrs, a_vals, mask=q_mask)


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

    TOTAL_BATCH = batch * seq_len
    BLOCK_B = 4 if TOTAL_BATCH >= 4 else 2

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