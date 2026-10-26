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
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    qkvz_base = mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T
    qkv_base = mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T
    z_base = z + i_bs * NUM_HEADS_V * HEAD_V
    ba_base = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
    b_base = b + i_bs * NUM_HEADS_V
    a_base = a + i_bs * NUM_HEADS_V
    q_end: tl.constexpr = HEAD_QK
    k_end: tl.constexpr = q_end + HEAD_QK
    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V
    q_vals = tl.load(qkvz_base + tl.arange(0, q_end))
    k_vals = tl.load(qkvz_base + tl.arange(q_end, k_end))
    v_vals = tl.load(qkvz_base + tl.arange(k_end, v_end))
    z_vals = tl.load(qkvz_base + tl.arange(v_end, z_end))
    qkv_q_offset = i_qk * HEAD_QK
    qkv_k_offset = NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK
    qkv_v_offset = NUM_HEADS_QK * HEAD_QK * 2 + i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    tl.store(qkv_base + qkv_q_offset + tl.arange(0, HEAD_QK), q_vals)
    tl.store(qkv_base + qkv_k_offset + tl.arange(0, HEAD_QK), k_vals)
    tl.store(qkv_base + qkv_v_offset + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), v_vals)
    z_offset = i_qk * HEAD_V * NUM_HEADS_V_PER_QK
    tl.store(z_base + z_offset + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK), z_vals)
    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK
    b_vals = tl.load(ba_base + tl.arange(0, b_end))
    a_vals = tl.load(ba_base + tl.arange(b_end, a_end))
    ba_offset = i_qk * NUM_HEADS_V_PER_QK
    tl.store(b_base + ba_offset + tl.arange(0, NUM_HEADS_V_PER_QK), b_vals)
    tl.store(a_base + ba_offset + tl.arange(0, NUM_HEADS_V_PER_QK), a_vals)
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
    grid = (batch * seq_len, num_heads_qk)
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
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a
ALL_DECODER_LAYER_TYPES = {
    "attention": None,
    "linear_attention": None,
}
EntryClass = None