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
    HEADS_PER_PROG: tl.constexpr,
):
    i_full = tl.program_id(0)
    NUM_HEADS_GROUPS: tl.constexpr = (NUM_HEADS_QK + HEADS_PER_PROG - 1) // HEADS_PER_PROG
    i_bs_block = i_full // NUM_HEADS_GROUPS
    i_head_group = i_full % NUM_HEADS_GROUPS
    head_start = i_head_group * HEADS_PER_PROG

    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    NUM_HEADS_V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    V_CHUNK: tl.constexpr = HEAD_V * NUM_HEADS_V_PER_QK

    offs_h = tl.arange(0, HEADS_PER_PROG)
    offs_h_2d = offs_h[:, None]
    offs_q = tl.arange(0, HEAD_QK)
    offs_k = tl.arange(0, HEAD_QK)
    offs_v = tl.arange(0, V_CHUNK)
    offs_b = tl.arange(0, NUM_HEADS_V_PER_QK)

    for b_offset in range(BLOCK_B):
        b_idx = i_bs_block * BLOCK_B + b_offset
        if b_idx < TOTAL_BATCH:
            base_qkvz = mixed_qkvz + b_idx * NUM_HEADS_QK * QKVZ_DIM_T + head_start * QKVZ_DIM_T
            base_ba = mixed_ba + b_idx * NUM_HEADS_QK * BA_DIM_T + head_start * BA_DIM_T

            out_q_base = mixed_qkv + b_idx * NUM_HEADS_QK * QKV_DIM_T
            out_k_base = out_q_base + NUM_HEADS_QK * HEAD_QK
            out_v_base = out_q_base + NUM_HEADS_QK * HEAD_QK * 2
            z_base = z + b_idx * NUM_HEADS_V * HEAD_V + head_start * V_CHUNK
            b_base = b + b_idx * NUM_HEADS_V + head_start * NUM_HEADS_V_PER_QK
            a_base = a + b_idx * NUM_HEADS_V + head_start * NUM_HEADS_V_PER_QK

            if NUM_HEADS_QK % HEADS_PER_PROG == 0:
                q_ptrs = base_qkvz + offs_h_2d * QKVZ_DIM_T + offs_q[None, :]
                q_vals = tl.load(q_ptrs)
                tl.store(
                    out_q_base + head_start * HEAD_QK + offs_h_2d * HEAD_QK + offs_q[None, :],
                    q_vals,
                )

                k_ptrs = base_qkvz + offs_h_2d * QKVZ_DIM_T + HEAD_QK + offs_k[None, :]
                k_vals = tl.load(k_ptrs)
                tl.store(
                    out_k_base + head_start * HEAD_QK + offs_h_2d * HEAD_QK + offs_k[None, :],
                    k_vals,
                )

                v_ptrs = base_qkvz + offs_h_2d * QKVZ_DIM_T + HEAD_QK * 2 + offs_v[None, :]
                v_vals = tl.load(v_ptrs)
                tl.store(
                    out_v_base + head_start * V_CHUNK + offs_h_2d * V_CHUNK + offs_v[None, :],
                    v_vals,
                )

                z_ptrs = base_qkvz + offs_h_2d * QKVZ_DIM_T + HEAD_QK * 2 + V_CHUNK + offs_v[None, :]
                z_vals = tl.load(z_ptrs)
                tl.store(z_base + offs_h_2d * V_CHUNK + offs_v[None, :], z_vals)

                b_ptrs = base_ba + offs_h_2d * BA_DIM_T + offs_b[None, :]
                b_vals = tl.load(b_ptrs)
                tl.store(b_base + offs_h_2d * NUM_HEADS_V_PER_QK + offs_b[None, :], b_vals)

                a_ptrs = base_ba + offs_h_2d * BA_DIM_T + NUM_HEADS_V_PER_QK + offs_b[None, :]
                a_vals = tl.load(a_ptrs)
                tl.store(a_base + offs_h_2d * NUM_HEADS_V_PER_QK + offs_b[None, :], a_vals)
            else:
                head_count = tl.minimum(HEADS_PER_PROG, NUM_HEADS_QK - head_start)
                mask_h = offs_h < head_count
                safe_h = tl.minimum(offs_h, head_count - 1)
                safe_h_2d = safe_h[:, None]
                mask_2d = mask_h[:, None]

                q_ptrs = base_qkvz + safe_h_2d * QKVZ_DIM_T + offs_q[None, :]
                q_vals = tl.load(q_ptrs, mask=mask_2d)
                tl.store(
                    out_q_base + head_start * HEAD_QK + safe_h_2d * HEAD_QK + offs_q[None, :],
                    q_vals,
                    mask=mask_2d,
                )

                k_ptrs = base_qkvz + safe_h_2d * QKVZ_DIM_T + HEAD_QK + offs_k[None, :]
                k_vals = tl.load(k_ptrs, mask=mask_2d)
                tl.store(
                    out_k_base + head_start * HEAD_QK + safe_h_2d * HEAD_QK + offs_k[None, :],
                    k_vals,
                    mask=mask_2d,
                )

                v_ptrs = base_qkvz + safe_h_2d * QKVZ_DIM_T + HEAD_QK * 2 + offs_v[None, :]
                v_vals = tl.load(v_ptrs, mask=mask_2d)
                tl.store(
                    out_v_base + head_start * V_CHUNK + safe_h_2d * V_CHUNK + offs_v[None, :],
                    v_vals,
                    mask=mask_2d,
                )

                z_ptrs = base_qkvz + safe_h_2d * QKVZ_DIM_T + HEAD_QK * 2 + V_CHUNK + offs_v[None, :]
                z_vals = tl.load(z_ptrs, mask=mask_2d)
                tl.store(
                    z_base + safe_h_2d * V_CHUNK + offs_v[None, :],
                    z_vals,
                    mask=mask_2d,
                )

                b_ptrs = base_ba + safe_h_2d * BA_DIM_T + offs_b[None, :]
                b_vals = tl.load(b_ptrs, mask=mask_2d)
                tl.store(
                    b_base + safe_h_2d * NUM_HEADS_V_PER_QK + offs_b[None, :],
                    b_vals,
                    mask=mask_2d,
                )

                a_ptrs = base_ba + safe_h_2d * BA_DIM_T + NUM_HEADS_V_PER_QK + offs_b[None, :]
                a_vals = tl.load(a_ptrs, mask=mask_2d)
                tl.store(
                    a_base + safe_h_2d * NUM_HEADS_V_PER_QK + offs_b[None, :],
                    a_vals,
                    mask=mask_2d,
                )
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
    HEADS_PER_PROG = 2
    TOTAL_BATCH = batch * seq_len
    num_head_groups = triton.cdiv(num_heads_qk, HEADS_PER_PROG)
    grid = (triton.cdiv(TOTAL_BATCH, BLOCK_B) * num_head_groups,)

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
        HEADS_PER_PROG=HEADS_PER_PROG,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a
ALL_DECODER_LAYER_TYPES = {
    "attention": None,
    "linear_attention": None,
}
EntryClass = None