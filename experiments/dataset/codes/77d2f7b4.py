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
    pack_meta,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    PACK_K: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_pack = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        seq_idx = tl.load(pack_meta + i_pack * 3).to(tl.int32)
        start_chunk = tl.load(pack_meta + i_pack * 3 + 1).to(tl.int32)
        valid_count = tl.load(pack_meta + i_pack * 3 + 2).to(tl.int32)
        bos = tl.load(cu_seqlens + seq_idx).to(tl.int32)
        eos = tl.load(cu_seqlens + seq_idx + 1).to(tl.int32)
        T = eos - bos
    else:
        start_chunk = i_pack
        valid_count = 1
        bos = i_b * T
        eos = i_b * T + T

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if USE_TMA:
        desc = tl.make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = tl.make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])

    for k in tl.static_range(PACK_K):
        if k < valid_count:
            i_t = start_chunk + k

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
                b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
                b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

            b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
            b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

            limit1 = tl.minimum(16, T - i_t * BT)
            for i in range(2, limit1):
                b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
                b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
                b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)

            limit2 = tl.minimum(32, T - i_t * BT)
            for i in range(18, limit2):
                b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
                b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
                b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

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
                    [i_t * BT + 0, 0],
                    b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"),
                )
                desc_o.store(
                    [i_t * BT + 16, 0],
                    b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"),
                )
                desc_o.store(
                    [i_t * BT + 16, 16],
                    b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"),
                )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [32]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape

    if cu_seqlens is not None:
        lengths = prepare_lens(cu_seqlens).tolist()
        PACK_K = 1
        pack_meta_rows = []
        for seq_idx, length in enumerate(lengths):
            cnt = (int(length) + BT - 1) // BT
            pack_count = (cnt + PACK_K - 1) // PACK_K
            for p in range(pack_count):
                start = p * PACK_K
                valid = min(PACK_K, cnt - start)
                pack_meta_rows.append((seq_idx, start, valid))
        if len(pack_meta_rows) == 0:
            pack_meta_rows.append((0, 0, 0))
        pack_meta = torch.tensor(
            pack_meta_rows, dtype=torch.int32, device=cu_seqlens.device
        )
        NP = len(pack_meta_rows)
        Ai = torch.zeros_like(A, dtype=output_dtype)
        merge_16x16_to_32x32_inverse_kernel[NP, B * H](
            A=A,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            pack_meta=pack_meta,
            T=T,
            H=H,
            BT=BT,
            PACK_K=PACK_K,
            USE_TMA=False,
            DOT_PRECISION="ieee",
        )
    else:
        PACK_K = 1
        NT = triton.cdiv(T, BT)
        pack_meta = torch.empty((0,), dtype=torch.int32, device=A.device)
        Ai = torch.zeros_like(A, dtype=output_dtype)
        merge_16x16_to_32x32_inverse_kernel[NT, B * H](
            A=A,
            Ai=Ai,
            cu_seqlens=None,
            pack_meta=pack_meta,
            T=T,
            H=H,
            BT=BT,
            PACK_K=PACK_K,
            USE_TMA=False,
            DOT_PRECISION="ieee",
        )

    return Ai