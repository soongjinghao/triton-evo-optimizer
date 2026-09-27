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
    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_A_22 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = tl.make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = tl.make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    # Loop-invariant base offset: i_t * BT * H * BT
    base_offset = i_t * BT * H * BT
    # Unroll first loop: for i in range(2, min(16, T - i_t * BT))
    # Max 14 iterations (2..15), mask for dynamic boundary
    n_iters_1 = min(16, T - i_t * BT)
    if n_iters_1 > 2:
        # i = 2
        i_val = 2
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 3:
        i_val = 3
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 4:
        i_val = 4
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 5:
        i_val = 5
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 6:
        i_val = 6
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 7:
        i_val = 7
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 8:
        i_val = 8
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 9:
        i_val = 9
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 10:
        i_val = 10
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 11:
        i_val = 11
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 12:
        i_val = 12
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 13:
        i_val = 13
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 14:
        i_val = 14
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    if n_iters_1 > 15:
        i_val = 15
        b_a_11 = -tl.load(A + base_offset + i_val * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i_val)[:, None], b_a_11, b_Ai_11)
    # Unroll second loop: for i in range(16 + 2, min(32, T - i_t * BT))
    # i starts at 18, max 14 iterations (18..31)
    n_iters_2 = min(32, T - i_t * BT)
    if n_iters_2 > 18:
        i_val = 18
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 19:
        i_val = 19
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 20:
        i_val = 20
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 21:
        i_val = 21
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 22:
        i_val = 22
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 23:
        i_val = 23
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 24:
        i_val = 24
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 25:
        i_val = 25
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 26:
        i_val = 26
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 27:
        i_val = 27
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 28:
        i_val = 28
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 29:
        i_val = 29
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 30:
        i_val = 30
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    if n_iters_2 > 31:
        i_val = 31
        b_a_22 = -tl.load(A + base_offset + i_val * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i_val - 16)[:, None], b_a_22, b_Ai_22)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
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
            Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
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