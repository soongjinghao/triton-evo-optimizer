import os
import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
)
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
    start_t_ptr,
    seq_len_ptr,
    CHUNK_PAIRS: tl.constexpr,
):
    i_t_pair, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n0 = tl.load(start_t_ptr + i_t_pair * 2).to(tl.int32)
        i_t0 = tl.load(seq_len_ptr + i_t_pair * 2).to(tl.int32)
        bos0 = tl.load(cu_seqlens + i_n0).to(tl.int32)
        eos0 = tl.load(cu_seqlens + i_n0 + 1).to(tl.int32)
        T0 = eos0 - bos0
        i_n1 = tl.load(start_t_ptr + i_t_pair * 2 + 1).to(tl.int32)
        i_t1 = tl.load(seq_len_ptr + i_t_pair * 2 + 1).to(tl.int32)
        bos1 = tl.load(cu_seqlens + i_n1).to(tl.int32)
        eos1 = tl.load(cu_seqlens + i_n1 + 1).to(tl.int32)
        T1 = eos1 - bos1
    else:
        bos0, eos0 = i_b * T, i_b * T + T
        i_t0 = i_t_pair * 2
        T0 = T
        bos1, eos1 = i_b * T, i_b * T + T
        i_t1 = i_t_pair * 2 + 1
        T1 = T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    for chunk_idx in range(2):
        if CHUNK_PAIRS == 1 and chunk_idx == 1:
            break
        if chunk_idx == 0:
            bos = bos0
            i_t = i_t0
            T_seq = T0
        else:
            bos = bos1
            i_t = i_t1
            T_seq = T1

        A_base = A + (bos * H + i_h) * BT
        Ai_base = Ai + (bos * H + i_h) * 16
        offset = (i_t * 16) % BT

        if not USE_TMA:
            p_A = tl.make_block_ptr(
                A_base, (T_seq, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
            )
            b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = None
            desc_o = None
            b_A = desc.load([i_t * 16, offset]).to(tl.float32)

        b_A = -tl.where(m_A, b_A, 0)

        for i in range(2, min(16, T_seq - i_t * 16)):
            b_a = -tl.load(A_base + (i_t * 16 + i) * H * BT + o_i + offset)
            b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
            b_A = tl.where((o_i == i)[:, None], b_a, b_A)

        b_A += m_I

        if not USE_TMA:
            p_Ai = tl.make_block_ptr(
                Ai_base, (T_seq, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0)
            )
            tl.store(
                p_Ai,
                b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )
        else:
            desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


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


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [16]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        start_t = chunk_indices[:, 0].contiguous()
        seq_len = chunk_indices[:, 1].contiguous()
        NT_pairs = (NT + 1) // 2
        start_t_pairs = torch.zeros(NT_pairs * 2, dtype=torch.int32, device=A.device)
        seq_len_pairs = torch.zeros(NT_pairs * 2, dtype=torch.int32, device=A.device)
        start_t_pairs[:NT] = start_t
        seq_len_pairs[:NT] = seq_len
        if NT % 2 == 1:
            start_t_pairs[NT] = start_t[-1].item()
            seq_len_pairs[NT] = seq_len[-1].item()
        CHUNK_PAIRS = 2 if NT >= 32 else 1
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)
        NT_pairs = (NT + 1) // 2
        start_t_pairs = None
        seq_len_pairs = None
        CHUNK_PAIRS = 2 if NT >= 32 else 1

    Ai = torch.zeros_like(A, dtype=output_dtype)

    if CHUNK_PAIRS == 2:
        solve_tril_16x16_kernel[NT_pairs, B * H](
            A=A,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            USE_TMA=False,
            DOT_PRECISION=False,
            start_t_ptr=start_t_pairs,
            seq_len_ptr=seq_len_pairs,
            CHUNK_PAIRS=CHUNK_PAIRS,
        )
    else:
        solve_tril_16x16_kernel[NT, B * H](
            A=A,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            USE_TMA=False,
            DOT_PRECISION=False,
            start_t_ptr=chunk_indices[:, 0].contiguous() if cu_seqlens is not None else None,
            seq_len_ptr=chunk_indices[:, 1].contiguous() if cu_seqlens is not None else None,
            CHUNK_PAIRS=1,
        )

    return Ai