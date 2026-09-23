import os
import torch
import triton
import triton.language as tl

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit
def merge_16x16_to_32x32_inverse_kernel(
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT
    chunk_start = i_t * BT
    row_stride = H * BT
    chunk_row_off = chunk_start * row_stride
    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(
            A, (T, BT), (row_stride, 1), (chunk_start, 0), (16, 16), (1, 0)
        )
        p_A_22 = tl.make_block_ptr(
            A, (T, BT), (row_stride, 1), (chunk_start + 16, 16), (16, 16), (1, 0)
        )
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = tl.make_tensor_descriptor(A, [T, BT], [row_stride, 1], [16, 16])
        desc_o = tl.make_tensor_descriptor(Ai, [T, BT], [row_stride, 1], [16, 16])
        b_Ai_11 = desc.load([chunk_start + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([chunk_start + 16, 16]).to(tl.float32)
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    limit1 = tl.minimum(16, T - chunk_start)
    for i in range(2, limit1):
        b_a_11 = -tl.load(A + chunk_row_off + i * row_stride + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    limit2 = tl.minimum(32, T - chunk_start)
    for i in range(18, limit2):
        b_a_22 = -tl.load(A + chunk_row_off + i * row_stride + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(
            A, (T, BT), (row_stride, 1), (chunk_start + 16, 0), (16, 16), (1, 0)
        )
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([chunk_start + 16, 0]).to(tl.float32)
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(
            Ai, (T, BT), (row_stride, 1), (chunk_start, 0), (16, 16), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai, (T, BT), (row_stride, 1), (chunk_start + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai, (T, BT), (row_stride, 1), (chunk_start + 16, 16), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store(
            [chunk_start + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [chunk_start + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [chunk_start + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )

def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [32]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    merge_16x16_to_32x32_inverse_kernel[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=False,
        DOT_PRECISION="ieee",
    )
    return Ai