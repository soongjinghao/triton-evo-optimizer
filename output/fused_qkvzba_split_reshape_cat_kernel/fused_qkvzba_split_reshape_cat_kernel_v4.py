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
    i_qk = i_full - i_bs_block * NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    stride_qkvz = NUM_HEADS_QK * QKVZ_DIM_T
    stride_qkv = NUM_HEADS_QK * QKV_DIM_T
    stride_z = NUM_HEADS_V * HEAD_V
    stride_ba = NUM_HEADS_QK * BA_DIM_T
    stride_b = NUM_HEADS_V
    stride_a = NUM_HEADS_V
    start_b_idx = i_bs_block * BLOCK_B
    pt_qkvz = mixed_qkvz + start_b_idx * stride_qkvz + i_qk * QKVZ_DIM_T
    pt_q = mixed_qkv + start_b_idx * stride_qkv + i_qk * HEAD_QK
    pt_k = mixed_qkv + start_b_idx * stride_qkv + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
    pt_v = mixed_qkv + start_b_idx * stride_qkv + NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    pt_z = z + start_b_idx * stride_z + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    pt_ba = mixed_ba + start_b_idx * stride_ba + i_qk * BA_DIM_T
    pt_b = b + start_b_idx * stride_b + i_qk * NUM_HEADS_V_PER_QK
    pt_a = a + start_b_idx * stride_a + i_qk * NUM_HEADS_V_PER_QK
    q_end: tl.constexpr = HEAD_QK
    k_end: tl.constexpr = q_end + HEAD_QK
    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V
    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK
    for b_offset in range(BLOCK_B):
        b_idx = start_b_idx + b_offset
        if b_idx < TOTAL_BATCH:
            q_vals = tl.load(pt_qkvz + tl.arange(0, q_end))
            k_vals = tl.load(pt_qkvz + tl.arange(q_end, k_end))
            v_vals = tl.load(pt_qkvz + tl.arange(k_end, v_end))
            z_vals = tl.load(pt_qkvz + tl.arange(v_end, z_end))
            tl.store(pt_q + tl.arange(0, HEAD_QK), q_vals)
            tl.store(pt_k + tl.arange(0, HEAD_QK), k_vals)
            tl.store(pt_v + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)
            tl.store(pt_z + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)
            b_vals = tl.load(pt_ba + tl.arange(0, b_end))
            a_vals = tl.load(pt_ba + tl.arange(b_end, a_end))
            tl.store(pt_b + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
            tl.store(pt_a + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)
        pt_qkvz += stride_qkvz
        pt_q += stride_qkv
        pt_k += stride_qkv
        pt_v += stride_qkv
        pt_z += stride_z
        pt_ba += stride_ba
        pt_b += stride_b
        pt_a += stride_a
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