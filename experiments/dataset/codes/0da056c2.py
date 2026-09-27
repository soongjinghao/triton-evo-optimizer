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
    i_t_packed, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t_base = (
            tl.load(chunk_indices + i_t_packed * 4).to(tl.int32),
            tl.load(chunk_indices + i_t_packed * 4 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_base = i_t_packed * 2

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Process tile 0
    if not USE_TMA:
        p_A_11_0 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_base * BT, 0), (16, 16), (1, 0)
        )
        p_A_22_0 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_base * BT + 16, 16), (16, 16), (1, 0)
        )
        b_Ai_11_0 = tl.load(p_A_11_0, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22_0 = tl.load(p_A_22_0, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = tl.make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = tl.make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11_0 = desc.load([i_t_base * BT + 0, 0]).to(tl.float32)
        b_Ai_22_0 = desc.load([i_t_base * BT + 16, 16]).to(tl.float32)

    b_Ai_11_0 = -tl.where(m_A, b_Ai_11_0, 0)
    b_Ai_22_0 = -tl.where(m_A, b_Ai_22_0, 0)

    for i in range(2, min(16, T - i_t_base * BT)):
        b_a_11_0 = -tl.load(A + (i_t_base * BT + i) * H * BT + o_i)
        b_a_11_0 += tl.sum(b_a_11_0[:, None] * b_Ai_11_0, 0)
        b_Ai_11_0 = tl.where((o_i == i)[:, None], b_a_11_0, b_Ai_11_0)

    for i in range(16 + 2, min(32, T - i_t_base * BT)):
        b_a_22_0 = -tl.load(A + (i_t_base * BT + i) * H * BT + o_i + 16)
        b_a_22_0 += tl.sum(b_a_22_0[:, None] * b_Ai_22_0, 0)
        b_Ai_22_0 = tl.where((o_i == i - 16)[:, None], b_a_22_0, b_Ai_22_0)

    b_Ai_11_0 += m_I
    b_Ai_22_0 += m_I

    if not USE_TMA:
        p_A_21_0 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_base * BT + 16, 0), (16, 16), (1, 0)
        )
        b_A_21_0 = tl.load(p_A_21_0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21_0 = desc.load([i_t_base * BT + 16, 0]).to(tl.float32)

    b_Ai_21_0 = -tl.dot(
        tl.dot(b_Ai_22_0, b_A_21_0, input_precision=DOT_PRECISION),
        b_Ai_11_0,
        input_precision=DOT_PRECISION,
    )

    # Process tile 1
    i_t_1 = i_t_base + 1
    if not USE_TMA:
        p_A_11_1 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_1 * BT, 0), (16, 16), (1, 0)
        )
        p_A_22_1 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_1 * BT + 16, 16), (16, 16), (1, 0)
        )
        b_Ai_11_1 = tl.load(p_A_11_1, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22_1 = tl.load(p_A_22_1, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_Ai_11_1 = desc.load([i_t_1 * BT + 0, 0]).to(tl.float32)
        b_Ai_22_1 = desc.load([i_t_1 * BT + 16, 16]).to(tl.float32)

    b_Ai_11_1 = -tl.where(m_A, b_Ai_11_1, 0)
    b_Ai_22_1 = -tl.where(m_A, b_Ai_22_1, 0)

    for i in range(2, min(16, T - i_t_1 * BT)):
        b_a_11_1 = -tl.load(A + (i_t_1 * BT + i) * H * BT + o_i)
        b_a_11_1 += tl.sum(b_a_11_1[:, None] * b_Ai_11_1, 0)
        b_Ai_11_1 = tl.where((o_i == i)[:, None], b_a_11_1, b_Ai_11_1)

    for i in range(16 + 2, min(32, T - i_t_1 * BT)):
        b_a_22_1 = -tl.load(A + (i_t_1 * BT + i) * H * BT + o_i + 16)
        b_a_22_1 += tl.sum(b_a_22_1[:, None] * b_Ai_22_1, 0)
        b_Ai_22_1 = tl.where((o_i == i - 16)[:, None], b_a_22_1, b_Ai_22_1)

    b_Ai_11_1 += m_I
    b_Ai_22_1 += m_I

    if not USE_TMA:
        p_A_21_1 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t_1 * BT + 16, 0), (16, 16), (1, 0)
        )
        b_A_21_1 = tl.load(p_A_21_1, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21_1 = desc.load([i_t_1 * BT + 16, 0]).to(tl.float32)

    b_Ai_21_1 = -tl.dot(
        tl.dot(b_Ai_22_1, b_A_21_1, input_precision=DOT_PRECISION),
        b_Ai_11_1,
        input_precision=DOT_PRECISION,
    )

    # Store tile 0
    if not USE_TMA:
        p_Ai_11_0 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_base * BT, 0), (16, 16), (1, 0)
        )
        p_Ai_21_0 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_base * BT + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_22_0 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_base * BT + 16, 16), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai_11_0,
            b_Ai_11_0.to(p_Ai_11_0.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22_0,
            b_Ai_22_0.to(p_Ai_22_0.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21_0,
            b_Ai_21_0.to(p_Ai_21_0.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store(
            [i_t_base * BT + 0, 0], b_Ai_11_0.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t_base * BT + 16, 0], b_Ai_21_0.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t_base * BT + 16, 16], b_Ai_22_0.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )

    # Store tile 1
    if not USE_TMA:
        p_Ai_11_1 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_1 * BT, 0), (16, 16), (1, 0)
        )
        p_Ai_21_1 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_1 * BT + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_22_1 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t_1 * BT + 16, 16), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai_11_1,
            b_Ai_11_1.to(p_Ai_11_1.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22_1,
            b_Ai_22_1.to(p_Ai_22_1.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21_1,
            b_Ai_21_1.to(p_Ai_21_1.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store(
            [i_t_1 * BT + 0, 0], b_Ai_11_1.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t_1 * BT + 16, 0], b_Ai_21_1.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t_1 * BT + 16, 16], b_Ai_22_1.to(desc_o.dtype, fp_downcast_rounding="rtne")
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
    # Pack 2 tiles per program, handle odd NT with ceil division
    NT_packed = (NT + 1) // 2
    merge_16x16_to_32x32_inverse_kernel[NT_packed, B * H](
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