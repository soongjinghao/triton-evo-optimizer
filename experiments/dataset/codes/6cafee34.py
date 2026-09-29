import torch
import numpy as np
import triton
from typing import Optional
import os
import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_bh_t = tl.program_id(0)
    i_bh = i_bh_t // (BT // 16)
    i_t = i_bh_t % (BT // 16)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16).to(tl.int32)
    m_A = o_i[:, None].to(tl.float32) > o_i[None, :].to(tl.float32)
    m_I = o_i[:, None].to(tl.float32) == o_i[None, :].to(tl.float32)
    A_ptr = A + (bos * H + i_h) * BT
    Ai_ptr = Ai + (bos * H + i_h) * 16
    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(
            A_ptr, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = tl.make_tensor_descriptor(A_ptr, [T, BT], [H * BT, 1], [16, 16])
        desc_o = tl.make_tensor_descriptor(Ai_ptr, [T, 16], [H * 16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
    b_A = -tl.where(m_A.to(tl.int1), b_A, 0.0)
    n_rows = T - i_t * 16
    n_rows = tl.minimum(n_rows, 16)
    n_rows_even = n_rows & ~1
    for i in range(1, n_rows_even, 2):
        a_ptr_i = A_ptr + (i_t * 16 + i) * H * BT + tl.arange(0, 16) + offset
        b_a_i = -tl.load(a_ptr_i, mask=tl.arange(0, 16) < n_rows, other=0.0)
        b_a_i_expanded = b_a_i[:, None]
        product_i = b_a_i_expanded * b_A
        sum_result_i = tl.sum(product_i, 0)
        b_a_i = b_a_i + sum_result_i
        a_ptr_ip1 = A_ptr + (i_t * 16 + i + 1) * H * BT + tl.arange(0, 16) + offset
        b_a_ip1 = -tl.load(a_ptr_ip1, mask=tl.arange(0, 16) < n_rows, other=0.0)
        b_a_ip1_expanded = b_a_ip1[:, None]
        product_ip1 = b_a_ip1_expanded * b_A
        sum_result_ip1 = tl.sum(product_ip1, 0)
        b_a_ip1 = b_a_ip1 + sum_result_ip1
        mask_i = (o_i == i)[:, None]
        mask_ip1 = (o_i == i + 1)[:, None]
        b_A = tl.where(mask_i | mask_ip1, tl.where(mask_i, b_a_i, b_a_ip1), b_A)
    if n_rows_even < n_rows:
        i = n_rows_even
        a_ptr = A_ptr + (i_t * 16 + i) * H * BT + tl.arange(0, 16) + offset
        b_a = -tl.load(a_ptr, mask=tl.arange(0, 16) < n_rows, other=0.0)
        b_a_expanded = b_a[:, None]
        product = b_a_expanded * b_A
        sum_result = tl.sum(product, 0)
        b_a = b_a + sum_result
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I.to(tl.float32)
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(
            Ai_ptr, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))

def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [16]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    grid = (B * H * (BT // 16),)
    solve_tril_16x16_kernel[grid](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=False,
        DOT_PRECISION=False,
    )
    return Ai