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
    BLOCK_HEADS: tl.constexpr,
):
    i_bs = tl.program_id(0)
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V_PER_QK * HEAD_V
    
    head_offsets = tl.arange(0, BLOCK_HEADS)
    head_mask = head_offsets < (NUM_HEADS_QK - (NUM_HEADS_QK // BLOCK_HEADS) * BLOCK_HEADS + BLOCK_HEADS) if NUM_HEADS_QK % BLOCK_HEADS != 0 else head_offsets < BLOCK_HEADS
    head_mask = head_mask & (head_offsets < NUM_HEADS_QK)
    
    q_end: tl.constexpr = HEAD_QK
    k_end: tl.constexpr = q_end + HEAD_QK
    v_end: tl.constexpr = k_end + NUM_HEADS_V_PER_QK * HEAD_V
    z_end: tl.constexpr = v_end + NUM_HEADS_V_PER_QK * HEAD_V
    
    base_ptr = mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
    q_vals = tl.load(base_ptr + head_offsets[:, None] * QKVZ_DIM_T + tl.arange(0, q_end)[None, :], mask=head_mask[:, None])
    k_vals = tl.load(base_ptr + head_offsets[:, None] * QKVZ_DIM_T + tl.arange(q_end, k_end)[None, :], mask=head_mask[:, None])
    v_vals = tl.load(base_ptr + head_offsets[:, None] * QKVZ_DIM_T + tl.arange(k_end, v_end)[None, :], mask=head_mask[:, None])
    z_vals = tl.load(base_ptr + head_offsets[:, None] * QKVZ_DIM_T + tl.arange(v_end, z_end)[None, :], mask=head_mask[:, None])
    
    qkv_base = mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T
    blk_q_st_ptr = qkv_base + head_offsets[:, None] * HEAD_QK + tl.arange(0, HEAD_QK)[None, :]
    tl.store(blk_q_st_ptr, q_vals, mask=head_mask[:, None])
    
    blk_k_st_ptr = qkv_base + NUM_HEADS_QK * HEAD_QK + head_offsets[:, None] * HEAD_QK + tl.arange(0, HEAD_QK)[None, :]
    tl.store(blk_k_st_ptr, k_vals, mask=head_mask[:, None])
    
    blk_v_st_ptr = qkv_base + NUM_HEADS_QK * HEAD_QK * 2 + head_offsets[:, None] * HEAD_V * NUM_HEADS_V_PER_QK + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK)[None, :]
    tl.store(blk_v_st_ptr, v_vals, mask=head_mask[:, None])
    
    z_base = z + i_bs * NUM_HEADS_V * HEAD_V
    blk_z_st_ptr = z_base + head_offsets[:, None] * HEAD_V * NUM_HEADS_V_PER_QK + tl.arange(0, HEAD_V * NUM_HEADS_V_PER_QK)[None, :]
    tl.store(blk_z_st_ptr, z_vals, mask=head_mask[:, None])
    
    ba_base_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T
    b_end: tl.constexpr = NUM_HEADS_V_PER_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V_PER_QK
    b_vals = tl.load(ba_base_ptr + head_offsets[:, None] * BA_DIM_T + tl.arange(0, b_end)[None, :], mask=head_mask[:, None])
    a_vals = tl.load(ba_base_ptr + head_offsets[:, None] * BA_DIM_T + tl.arange(b_end, a_end)[None, :], mask=head_mask[:, None])
    
    ba_out_base = b + i_bs * NUM_HEADS_V
    blk_b_st_ptr = ba_out_base + head_offsets[:, None] * NUM_HEADS_V_PER_QK + tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]
    blk_a_st_ptr = a + i_bs * NUM_HEADS_V + head_offsets[:, None] * NUM_HEADS_V_PER_QK + tl.arange(0, NUM_HEADS_V_PER_QK)[None, :]
    tl.store(blk_b_st_ptr, b_vals, mask=head_mask[:, None])
    tl.store(blk_a_st_ptr, a_vals, mask=head_mask[:, None])

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
    
    BLOCK_HEADS = 4
    grid = (batch * seq_len,)
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
        BLOCK_HEADS,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a

ALL_DECODER_LAYER_TYPES = {
    "attention": None,
    "linear_attention": None,
}
EntryClass = None