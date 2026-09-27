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

    start_b_idx = i_bs_block * BLOCK_B
    b_idx = start_b_idx

    # mixed_qkvz source running pointer
    mz_ptr = mixed_qkvz + b_idx * (NUM_HEADS_QK * QKVZ_DIM_T) + i_qk * QKVZ_DIM_T

    # mixed_qkv dest q/k/v running pointers
    q_ptr = mixed_qkv + b_idx * (NUM_HEADS_QK * QKV_DIM_T) + i_qk * HEAD_QK
    k_ptr = mixed_qkv + b_idx * (NUM_HEADS_QK * QKV_DIM_T) + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
    v_ptr = mixed_qkv + b_idx * (NUM_HEADS_QK * QKV_DIM_T) + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK

    z_ptr = z + b_idx * (NUM_HEADS_V * HEAD_V) + i_qk * HEAD_V * NUM_HEADS_V_PER_QK

    ba_ptr = mixed_ba + b_idx * (NUM_HEADS_QK * BA_DIM_T) + i_qk * BA_DIM_T

    b_ptr = b + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK
    a_ptr = a + b_idx * NUM_HEADS_V + i_qk * NUM_HEADS_V_PER_QK

    for b_offset in range(BLOCK_B):
        if b_idx < TOTAL_BATCH:
            q_vals = tl.load(mz_ptr + tl.arange(0, q_end))
            k_vals = tl.load(mz_ptr + tl.arange(q_end, k_end))
            v_vals = tl.load(mz_ptr + tl.arange(k_end, v_end))
            z_vals = tl.load(mz_ptr + tl.arange(v_end, z_end))

            tl.store(q_ptr + tl.arange(0, HEAD_QK), q_vals)
            tl.store(k_ptr + tl.arange(0, HEAD_QK), k_vals)
            tl.store(v_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)

            tl.store(z_ptr + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)

            b_vals = tl.load(ba_ptr + tl.arange(0, b_end))
            a_vals = tl.load(ba_ptr + tl.arange(b_end, a_end))

            tl.store(b_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
            tl.store(a_ptr + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)

        # advance running pointers by one batch row stride
        b_idx = b_idx + 1
        mz_ptr = mz_ptr + NUM_HEADS_QK * QKVZ_DIM_T
        q_ptr = q_ptr + NUM_HEADS_QK * QKV_DIM_T
        k_ptr = k_ptr + NUM_HEADS_QK * QKV_DIM_T
        v_ptr = v_ptr + NUM_HEADS_QK * QKV_DIM_T
        z_ptr = z_ptr + NUM_HEADS_V * HEAD_V
        ba_ptr = ba_ptr + NUM_HEADS_QK * BA_DIM_T
        b_ptr = b_ptr + NUM_HEADS_V
        a_ptr = a_ptr + NUM_HEADS_V


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