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
    HEAD_GROUP: tl.constexpr,
):
    i_full = tl.program_id(0)
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    NUM_HEAD_GROUPS: tl.constexpr = NUM_HEADS_QK // HEAD_GROUP
    TOTAL_BS_BLOCKS: tl.constexpr = (TOTAL_BATCH + BLOCK_B - 1) // BLOCK_B

    i_bs_block = i_full // NUM_HEAD_GROUPS
    i_head_group = i_full % NUM_HEAD_GROUPS

    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V

    head_qk_start = i_head_group * HEAD_GROUP

    off_h = tl.arange(0, HEAD_GROUP)[:, None]
    off_hq = tl.arange(0, HEAD_QK)[None, :]
    off_hv = tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK)[None, :]
    off_bcol = tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]

    for b_offset in range(BLOCK_B):
        b_idx = i_bs_block * BLOCK_B + b_offset
        if b_idx < TOTAL_BATCH:
            mixed_qkvz_batch_base = (
                mixed_qkvz
                + b_idx * NUM_HEADS_QK * QKVZ_DIM_T
                + head_qk_start * QKVZ_DIM_T
            )
            q_in_ptrs = mixed_qkvz_batch_base + off_h * QKVZ_DIM_T + off_hq
            k_in_ptrs = mixed_qkvz_batch_base + HEAD_QK + off_h * QKVZ_DIM_T + off_hq
            v_in_ptrs = mixed_qkvz_batch_base + 2 * HEAD_QK + off_h * QKVZ_DIM_T + off_hv
            z_in_ptrs = (
                mixed_qkvz_batch_base
                + 2 * HEAD_QK
                + NUM_HEADS_V_PER_QK * HEAD_V
                + off_h * QKVZ_DIM_T
                + off_hv
            )

            q_vals = tl.load(q_in_ptrs)
            k_vals = tl.load(k_in_ptrs)
            v_vals = tl.load(v_in_ptrs)
            z_vals = tl.load(z_in_ptrs)

            mixed_qkv_batch_base = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T
            q_base = mixed_qkv_batch_base + head_qk_start * HEAD_QK
            k_base = mixed_qkv_batch_base + NUM_HEADS_QK * HEAD_QK + head_qk_start * HEAD_QK
            v_base = (
                mixed_qkv_batch_base
                + NUM_HEADS_QK * HEAD_QK * 2
                + head_qk_start * (NUM_HEADS_V_PER_QK * HEAD_V)
            )

            q_out_ptrs = q_base + off_h * HEAD_QK + off_hq
            k_out_ptrs = k_base + off_h * HEAD_QK + off_hq
            v_out_ptrs = v_base + off_h * (NUM_HEADS_V_PER_QK * HEAD_V) + off_hv

            tl.store(q_out_ptrs, q_vals)
            tl.store(k_out_ptrs, k_vals)
            tl.store(v_out_ptrs, v_vals)

            z_base = z + b_idx * NUM_HEADS_V * HEAD_V + head_qk_start * (NUM_HEADS_V_PER_QK * HEAD_V)
            z_out_ptrs = z_base + off_h * (NUM_HEADS_V_PER_QK * HEAD_V) + off_hv
            tl.store(z_out_ptrs, z_vals)

            mixed_ba_batch_base = (
                mixed_ba
                + b_idx * NUM_HEADS_QK * BA_DIM_T
                + head_qk_start * BA_DIM_T
            )
            b_in_ptrs = mixed_ba_batch_base + off_h * BA_DIM_T + off_bcol
            a_in_ptrs = mixed_ba_batch_base + NUM_HEADS_V_PER_QK + off_h * BA_DIM_T + off_bcol

            b_vals = tl.load(b_in_ptrs)
            a_vals = tl.load(a_in_ptrs)

            b_base = b + b_idx * NUM_HEADS_V + head_qk_start * NUM_HEADS_V_PER_QK
            a_base = a + b_idx * NUM_HEADS_V + head_qk_start * NUM_HEADS_V_PER_QK

            b_out_ptrs = b_base + off_h * NUM_HEADS_V_PER_QK + off_bcol
            a_out_ptrs = a_base + off_h * NUM_HEADS_V_PER_QK + off_bcol

            tl.store(b_out_ptrs, b_vals)
            tl.store(a_out_ptrs, a_vals)
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
    if num_heads_qk % 2 == 0:
        head_group = 2
    else:
        head_group = 1
    grid = (triton.cdiv(TOTAL_BATCH, BLOCK_B) * (num_heads_qk // head_group),)
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
        HEAD_GROUP=head_group,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a
ALL_DECODER_LAYER_TYPES = {
    "attention": None,
    "linear_attention": None,
}
EntryClass = None