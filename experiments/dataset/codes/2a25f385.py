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
    K_PACK: tl.constexpr,
):
    i_t_pack, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t_base = (
            tl.load(chunk_indices + i_t_pack * 2).to(tl.int32),
            tl.load(chunk_indices + i_t_pack * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_base = i_t_pack * K_PACK
    
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    
    A_base = A + (bos * H + i_h) * BT
    Ai_base = Ai + (bos * H + i_h) * 16
    
    n_blocks = tl.cdiv(T, 16)
    n_full_packs = n_blocks // K_PACK
    is_tail = i_t_pack >= n_full_packs
    k_actual = K_PACK - (K_PACK - 1) * is_tail.to(tl.int32)
    
    if is_tail:
        k_actual = n_blocks - n_full_packs * K_PACK
        if k_actual <= 0:
            return
    
    offset_base = (i_t_base * 16) % BT
    
    if k_actual == 1:
        offset = offset_base
        if not USE_TMA:
            p_A = tl.make_block_ptr(
                A_base, (T, BT), (H * BT, 1), (i_t_base * 16, offset), (16, 16), (1, 0)
            )
            b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = None
            desc_o = None
            b_A = desc.load([i_t_base * 16, offset]).to(tl.float32)
        b_A = -tl.where(m_A, b_A, 0)
        for i in range(2, min(16, T - i_t_base * 16)):
            b_a = -tl.load(A_base + (i_t_base * 16 + i) * H * BT + o_i + offset)
            b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
            b_A = tl.where((o_i == i)[:, None], b_a, b_A)
        b_A += m_I
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(
                Ai_base, (T, 16), (H * 16, 1), (i_t_base * 16, 0), (16, 16), (1, 0)
            )
            tl.store(
                p_Ai,
                b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )
        else:
            desc_o.store([i_t_base * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    else:
        offset_0 = offset_base
        offset_1 = (offset_base + 16) % BT
        if not USE_TMA:
            p_A_0 = tl.make_block_ptr(
                A_base, (T, BT), (H * BT, 1), (i_t_base * 16, offset_0), (16, 16), (1, 0)
            )
            p_A_1 = tl.make_block_ptr(
                A_base, (T, BT), (H * BT, 1), ((i_t_base + 1) * 16, offset_1), (16, 16), (1, 0)
            )
            b_A_0 = tl.load(p_A_0, boundary_check=(0, 1)).to(tl.float32)
            b_A_1 = tl.load(p_A_1, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = None
            desc_o = None
            b_A_0 = desc.load([i_t_base * 16, offset_0]).to(tl.float32)
            b_A_1 = desc.load([(i_t_base + 1) * 16, offset_1]).to(tl.float32)
        
        b_A_0 = -tl.where(m_A, b_A_0, 0)
        b_A_1 = -tl.where(m_A, b_A_1, 0)
        
        for i in range(2, min(16, T - i_t_base * 16)):
            b_a_0 = -tl.load(A_base + (i_t_base * 16 + i) * H * BT + o_i + offset_0)
            b_a_0 = b_a_0 + tl.sum(b_a_0[:, None] * b_A_0, 0)
            b_A_0 = tl.where((o_i == i)[:, None], b_a_0, b_A_0)
        
        for i in range(2, min(16, T - (i_t_base + 1) * 16)):
            b_a_1 = -tl.load(A_base + ((i_t_base + 1) * 16 + i) * H * BT + o_i + offset_1)
            b_a_1 = b_a_1 + tl.sum(b_a_1[:, None] * b_A_1, 0)
            b_A_1 = tl.where((o_i == i)[:, None], b_a_1, b_A_1)
        
        b_A_0 += m_I
        b_A_1 += m_I
        
        if not USE_TMA:
            p_Ai_0 = tl.make_block_ptr(
                Ai_base, (T, 16), (H * 16, 1), (i_t_base * 16, 0), (16, 16), (1, 0)
            )
            p_Ai_1 = tl.make_block_ptr(
                Ai_base, (T, 16), (H * 16, 1), ((i_t_base + 1) * 16, 0), (16, 16), (1, 0)
            )
            tl.store(
                p_Ai_0,
                b_A_0.to(p_Ai_0.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )
            tl.store(
                p_Ai_1,
                b_A_1.to(p_Ai_1.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )
        else:
            desc_o.store([i_t_base * 16, 0], b_A_0.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([(i_t_base + 1) * 16, 0], b_A_1.to(desc_o.dtype, fp_downcast_rounding="rtne"))

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
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    K_PACK = 2
    NT_packed = triton.cdiv(NT, K_PACK)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    solve_tril_16x16_kernel[NT_packed, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=False,
        DOT_PRECISION=False,
        K_PACK=K_PACK,
    )
    return Ai