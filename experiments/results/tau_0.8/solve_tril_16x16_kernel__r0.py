import os
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
)
@triton.jit
def solve_tril_16x16_kernel_varlen(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_n, i_t_local = (
        tl.load(chunk_indices + i_t * 2).to(tl.int32),
        tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
    )
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int32),
        tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    )
    T_local = eos - bos
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_ptr = A + (bos * H + i_h) * BT
    Ai_ptr = Ai + (bos * H + i_h) * 16
    offset = (i_t_local * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(
            A_ptr, (T_local, BT), (H * BT, 1), (i_t_local * 16, offset), (16, 16), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = None
        desc_o = None
        b_A = desc.load([i_t_local * 16, offset]).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)
    for i in range(2, min(16, T_local - i_t_local * 16)):
        b_a = -tl.load(A_ptr + (i_t_local * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(
            Ai_ptr, (T_local, 16), (H * 16, 1), (i_t_local * 16, 0), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store([i_t_local * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
)
@triton.jit
def solve_tril_16x16_kernel_fixed(
    A,
    Ai,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_ptr = A + (bos * H + i_h) * BT
    Ai_ptr = Ai + (bos * H + i_h) * 16
    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(
            A_ptr, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = None
        desc_o = None
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)
    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A_ptr + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
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
        Ai = torch.zeros_like(A, dtype=output_dtype)
        solve_tril_16x16_kernel_varlen[NT, B * H](
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
    else:
        NT = triton.cdiv(T, BT)
        Ai = torch.zeros_like(A, dtype=output_dtype)
        solve_tril_16x16_kernel_fixed[NT, B * H](
            A=A,
            Ai=Ai,
            T=T,
            H=H,
            BT=BT,
            USE_TMA=False,
            DOT_PRECISION=False,
        )
    return Ai