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


def prepare_chunk_meta(
    cu_seqlens: torch.LongTensor, chunk_indices: torch.LongTensor
) -> torch.LongTensor:
    seq_idx = chunk_indices[:, 0]
    chunk_idx = chunk_indices[:, 1]
    bos = cu_seqlens[seq_idx]
    eos = cu_seqlens[seq_idx + 1]
    seq_len = eos - bos
    return torch.stack([seq_idx, chunk_idx, bos, eos, seq_len], 1).to(cu_seqlens)


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
    chunk_meta,
):
    pid_chunk = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_meta + pid_chunk * 5 + 0).to(tl.int32)
        i_t = tl.load(chunk_meta + pid_chunk * 5 + 1).to(tl.int32)
        bos = tl.load(chunk_meta + pid_chunk * 5 + 2).to(tl.int32)
        eos = tl.load(chunk_meta + pid_chunk * 5 + 3).to(tl.int32)
        T_len = tl.load(chunk_meta + pid_chunk * 5 + 4).to(tl.int32)
    else:
        i_t = pid_chunk
        bos = i_b * T
        eos = i_b * T + T
        T_len = T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    row_base = (i_t * BT) * H * BT + o_i
    col_base = row_base + 16

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(
            A, (T_len, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_A_22 = tl.make_block_ptr(
            A, (T_len, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        b_Ai