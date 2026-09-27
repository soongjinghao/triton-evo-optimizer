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
    i_t_group, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n_0, i_t_0 = (
            tl.load(chunk_indices + (i_t_group * 2) * 2).to(tl.int32),
            tl.load(chunk_indices + (i_t_group * 2) * 2 + 1).to(tl.int32),
        )
        bos_0, eos_0 = (
            tl.load(cu_seqlens + i_n_0).to(tl.int32),
            tl.load(cu_seqlens + i_n_0 + 1).to(tl.int32),
        )
        T_0 = eos_0 - bos_0
        i_n_1, i_t_1 = (
            tl.load(chunk_indices + (i_t_group * 2 + 1) * 2).to(tl.int32),
            tl.load(chunk_indices + (i_t_group * 2 + 1) * 2 + 1).to(tl.int32),
        )
        bos_1, eos_1 = (
            tl.load(cu_seqlens + i_n_1).to(tl.int32),
            tl.load(cu_seqlens + i_n_1 + 1).to(tl.int32),
        )
        T_1 = eos_1 - bos_1
    else:
        bos_0 = i_b * T
        eos_0 = i_b * T + T
        T_0 = T
        bos_1 = i_b * T
        eos_1 = i_b * T + T
        T_1 = T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A
    Ai_base = Ai
    for row_offset in range(2):
        if row_offset == 0:
            i_t = i_t_0 if IS_VARLEN else i_t_group * 2
            bos = bos_0
            eos = eos_0
            T_cur = T_0
        else:
            i_t = i_t_1 if IS_VARLEN else i_t_group * 2 + 1
            bos = bos_1
            eos = eos_1
            T_cur = T_1
        A = A_base + (bos * H + i_h) * BT
        Ai = Ai_base + (bos * H + i_h) * BT
        if not USE_TMA:
            p_A_11 = tl.make_block_ptr(
                A, (T_cur, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
            )
            p_A_22 = tl.make_block_ptr(
                A, (T_cur, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
            )
            b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = tl.make_tensor_descriptor(A, [T_cur, BT], [H * BT, 1], [16, 16])
            desc_o = tl.make_tensor_descriptor(Ai, [T_cur, BT], [H * BT, 1], [16, 16])
            b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
            b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
        b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
        for i in range(2, min(16, T_cur - i_t * BT)):
            b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
            b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
            b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
        for i in range(16 + 2, min(32, T_cur - i_t * BT)):
            b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
            b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
            b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
        b_Ai_11 += m_I
        b_Ai_22 += m_I
        if not USE_TMA:
            p_A_21 = tl.make_block_ptr(
                A, (T_cur, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
            )
            b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_Ai_21 = -tl.dot(
            tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
            b_Ai_11,
            input_precision=DOT_PRECISION,
        )
        if not USE_TMA:
            p_Ai_11 = tl.make_block_ptr(
                Ai, (T_cur, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
            )
            p_Ai_21 = tl.make_block_ptr(
                Ai, (T_cur, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
            )
            p_Ai_22 = tl.make_block_ptr(
                Ai, (T_cur, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
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
                [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
            )
            desc_o.store(
                [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
            )
            desc_o.store(
                [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
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
    if NT % 2 == 0:
        merge_16x16_to_32x32_inverse_kernel[NT // 2, B * H](
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
    else:
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