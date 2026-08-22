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
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    i_bh = pid // NT
    i_t = pid % NT
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
    if T - i_t * 16 >= 16:
        for i in range(2, 16):
            a_ptr = A_ptr + (i_t * 16 + i) * H * BT + tl.arange(0, 16) + offset
            b_a = -tl.load(a_ptr)
            b_a_expanded = b_a[:, None]
            product = b_a_expanded * b_A
            sum_result = tl.sum(product, 0)
            b_a = b_a + sum_result
            b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    else:
        for i in range(2, min(16, T - i_t * 16)):
            a_ptr = A_ptr + (i_t * 16 + i) * H * BT + tl.arange(0, 16) + offset
            b_a = -tl.load(a_ptr, mask=tl.arange(0, 16) < (T - i_t * 16), other=0.0)
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
    A = A.contiguous()
    grid = (B * H * NT,)
    solve_tril_16x16_kernel[grid](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=True,
        NT=NT,
        DOT_PRECISION=False,
    )
    return Ai