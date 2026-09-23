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
    b_offs = tl.arange(0, BLOCK_B)
    b_idx = i_bs_block * BLOCK_B + b_offs
    mask_b = b_idx < TOTAL_BATCH
    mask_2d = mask_b[:, None]
    q_width: tl.constexpr = HEAD_QK
    qkvz_base = (
        mixed_qkvz
        + b_idx[:, None] * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
    )
    q_ptr = qkvz_base + tl.arange(0, q_width)[None, :]
    q_vals = tl.load(q_ptr, mask=mask_2d)
    k_col_off: tl.constexpr = HEAD_QK
    k_width: tl.constexpr = HEAD_QK
    k_ptr = qkvz_base + k_col_off + tl.arange(0, k_width)[None, :]
    k_vals = tl.load(k_ptr, mask=mask_2d)
    v_col_off: tl.constexpr = HEAD_QK * 2
    v_width: tl.constexpr = NUM_HEADS_V_PER_QK * HEAD_V
    v_ptr = qkvz_base + v_col_off + tl.arange(0, v_width)[None, :]
    v_vals = tl.load(v_ptr, mask=mask_2d)
    z_col_off: tl.constexpr = v_col_off + v_width
    z_width: tl.constexpr = NUM_HEADS_V_PER_QK * HEAD_V
    z_ptr = qkvz_base + z_col_off + tl.arange(0, z_width)[None, :]
    z_vals = tl.load(z_ptr, mask=mask_2d)
    qkv_out_base = (
        mixed_qkv
        + b_idx[:, None] * NUM_HEADS_QK * QKV_DIM_T
    )
    q_out_ptr = (
        qkv_out_base
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)[None, :]
    )
    tl.store(q_out_ptr, q_vals, mask=mask_2d)
    k_out_ptr = (
        qkv_out_base
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)[None, :]
    )
    tl.store(k_out_ptr, k_vals, mask=mask_2d)
    v_out_ptr = (
        qkv_out_base
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
        + tl.arange(0, v_width)[None, :]
    )
    tl.store(v_out_ptr, v_vals, mask=mask_2d)
    z_out_ptr = (
        z
        + b_idx[:, None] * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
        + tl.arange(0, z_width)[None, :]
    )
    tl.store(z_out_ptr, z_vals, mask=mask_2d)
    b_width: tl.constexpr = NUM_HEADS_V_PER_QK
    a_col_off: tl.constexpr = b_width
    ba_base_ptr = (
        mixed_ba
        + b_idx[:, None] * NUM_HEADS_QK * BA_DIM_T
        + i_qk * BA_DIM_T
    )
    b_vals = tl.load(ba_base_ptr + tl.arange(0, b_width)[None, :], mask=mask_2d)
    a_vals = tl.load(
        ba_base_ptr + a_col_off + tl.arange(0, b_width)[None, :],
        mask=mask_2d,
    )
    b_out_ptr = (
        b
        + b_idx[:, None] * NUM_HEADS_V
        + i_qk * NUM_HEADS_V_PER_QK
        + tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]
    )
    a_out_ptr = (
        a
        + b_idx[:, None] * NUM_HEADS_V
        + i_qk * NUM_HEADS_V_PER_QK
        + tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]
    )
    tl.store(b_out_ptr, b_vals, mask=mask_2d)
    tl.store(a_out_ptr, a_vals, mask=mask_2d)
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