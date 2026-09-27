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
    NT: tl.constexpr,
):
    i_t_packed, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        bos, eos = 0, 0
        T_local = T
    else:
        bos, eos = i_b * T, i_b * T + T
        T_local = T
    
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    
    A_base = A + (bos * H + i_h) * BT
    Ai_base = Ai + (bos * H + i_h) * 16
    
    for pack_idx in range(4):
        i_t = i_t_packed * 4 + pack_idx
        
        if IS_VARLEN:
            if i_t >= NT:
                continue
            i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
            i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos_local = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos_local = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_local = eos_local - bos_local
            A_base_local = A + (bos_local * H + i_h) * BT
            Ai_base_local = Ai + (bos_local * H + i_h) * 16
        else:
            i_t_local = i_t
            A_base_local = A_base
            Ai_base_local = Ai_base
        
        if i_t_local * 16 >= T_local:
            continue
        
        offset = (i_t_local * 16) % BT
        
        if not USE_TMA:
            p_A = tl.make_block_ptr(
                A_base_local, (T_local, BT), (H * BT, 1), (i_t_local * 16, offset), (16, 16), (1, 0)
            )
            b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = None
            desc_o = None
            b_A = desc.load([i_t_local * 16, offset]).to(tl.float32)
        
        b_A = -tl.where(m_A, b_A, 0)
        
        max_i = min(16, T_local - i_t_local * 16)
        for i in range(2, max_i):
            b_a = -tl.load(A_base_local + (i_t_local * 16 + i) * H * BT + o_i + offset)
            b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
            b_A = tl.where((o_i == i)[:, None], b_a, b_A)
        
        b_A += m_I
        
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(
                Ai_base_local, (T_local, 16), (H * 16, 1), (i_t_local * 16, 0), (16, 16), (1, 0)
            )
            tl.store(
                p_Ai,
                b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )
        else:
            desc_o.store([i_t_local * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))

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
    NT_packed = triton.cdiv(NT, 4)
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
        NT=NT,
    )
    return Ai