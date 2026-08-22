import warnings
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


BS_LIST = [32, 64]


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    NT: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    BLOCK_NT: tl.constexpr,
    BLOCK_NS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_comb = tl.program_id(1)

    total_bh = B * H
    if pid_bh >= total_bh:
        return

    # compute mask once
    o_i = tl.arange(0, BT)
    o_i_f32 = o_i.to(tl.float32)
    mask_reverse = tl.where(o_i_f32[:, None] <= o_i_f32[None, :], 1.0, 0.0)
    mask_forward = tl.where(o_i_f32[:, None] >= o_i_f32[None, :], 1.0, 0.0)
    m_s = tl.where(REVERSE, mask_reverse, mask_forward)

    total_s_tiles = tl.cdiv(S, BS)
    s_tile_groups = tl.cdiv(total_s_tiles, BLOCK_NS)
    nt_groups = tl.cdiv(NT, BLOCK_NT) if NT > 0 else 1

    group_nt = pid_comb // s_tile_groups
    group_s = pid_comb % s_tile_groups

    nt_start = group_nt * BLOCK_NT
    nt_end = min(nt_start + BLOCK_NT, NT)
    s_start = group_s * BLOCK_NS
    s_end = min(s_start + BLOCK_NS, total_s_tiles)

    i_b = pid_bh // H
    i_h = pid_bh % H

    if IS_VARLEN:
        cu_seqlens_vec = tl.load(cu_seqlens + tl.arange(0, B + 1)).to(tl.int32)
        bos = cu_seqlens_vec[i_b]
        eos = cu_seqlens_vec[i_b + 1]
        T_curr = eos - bos

        offs_nt = nt_start + tl.arange(0, BLOCK_NT)
        mask_nt = offs_nt < NT
        i_n_vec = tl.load(chunk_indices + offs_nt * 2, mask=mask_nt, other=0).to(tl.int32)
        i_t_local_vec = tl.load(chunk_indices + offs_nt * 2 + 1, mask=mask_nt, other=0).to(tl.int32)

        for nt_offset in range(BLOCK_NT):
            if mask_nt[nt_offset]:
                i_t_local = i_t_local_vec[nt_offset]
                if i_t_local * BT < T_curr:
                    for s_offset in range(BLOCK_NS):
                        s_idx = s_start + s_offset
                        if s_idx < s_end:
                            i_s = s_idx
                            if i_s * BS < S:
                                if HEAD_FIRST:
                                    p_s = tl.make_block_ptr(
                                        s + (bos * H + i_h * T_curr) * S,
                                        (T_curr, S),
                                        (S, 1),
                                        (i_t_local * BT, i_s * BS),
                                        (BT, BS),
                                        (1, 0),
                                    )
                                    p_o = tl.make_block_ptr(
                                        o + (bos * H + i_h * T_curr) * S,
                                        (T_curr, S),
                                        (S, 1),
                                        (i_t_local * BT, i_s * BS),
                                        (BT, BS),
                                        (1, 0),
                                    )
                                else:
                                    p_s = tl.make_block_ptr(
                                        s + (bos * H + i_h) * S,
                                        (T_curr, S),
                                        (H * S, 1),
                                        (i_t_local * BT, i_s * BS),
                                        (BT, BS),
                                        (1, 0),
                                    )
                                    p_o = tl.make_block_ptr(
                                        o + (bos * H + i_h) * S,
                                        (T_curr, S),
                                        (H * S, 1),
                                        (i_t_local * BT, i_s * BS),
                                        (BT, BS),
                                        (1, 0),
                                    )
                                b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
                                b_o = tl.dot(m_s, b_s, allow_tf32=False)
                                tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    else:
        bos = i_b * T
        T_curr = T
        for nt_offset in range(BLOCK_NT):
            nt_idx = nt_start + nt_offset
            if nt_idx < nt_end:
                i_t = nt_idx
                for s_offset in range(BLOCK_NS):
                    s_idx = s_start + s_offset
                    if s_idx < s_end:
                        i_s = s_idx
                        if i_t * BT < T_curr and i_s * BS < S:
                            if HEAD_FIRST:
                                p_s = tl.make_block_ptr(
                                    s + (bos * H + i_h * T_curr) * S,
                                    (T_curr, S),
                                    (S, 1),
                                    (i_t * BT, i_s * BS),
                                    (BT, BS),
                                    (1, 0),
                                )
                                p_o = tl.make_block_ptr(
                                    o + (bos * H + i_h * T_curr) * S,
                                    (T_curr, S),
                                    (S, 1),
                                    (i_t * BT, i_s * BS),
                                    (BT, BS),
                                    (1, 0),
                                )
                            else:
                                p_s = tl.make_block_ptr(
                                    s + (bos * H + i_h) * S,
                                    (T_curr, S),
                                    (H * S, 1),
                                    (i_t * BT, i_s * BS),
                                    (BT, BS),
                                    (1, 0),
                                )
                                p_o = tl.make_block_ptr(
                                    o + (bos * H + i_h) * S,
                                    (T_curr, S),
                                    (H * S, 1),
                                    (i_t * BT, i_s * BS),
                                    (BT, BS),
                                    (1, 0),
                                )
                            b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
                            b_o = tl.dot(m_s, b_s, allow_tf32=False)
                            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    NT = (
        triton.cdiv(T, BT)
        if cu_seqlens is None
        else (len(chunk_indices) if chunk_indices is not None else 0)
    )
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    BLOCK_BH = 4
    BLOCK_NT = 2
    BLOCK_NS = 2
    BS_min = min(BS_LIST)
    total_s_tiles_max = triton.cdiv(S, BS_min)
    s_tile_groups = triton.cdiv(total_s_tiles_max, BLOCK_NS)
    nt_groups = triton.cdiv(NT, BLOCK_NT) if NT > 0 else 1
    grid_0 = B * H
    grid_1 = nt_groups * s_tile_groups
    grid = (grid_0, max(grid_1, 1))

    chunk_local_cumsum_vector_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        NT=NT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        BLOCK_BH=BLOCK_BH,
        BLOCK_NT=BLOCK_NT,
        BLOCK_NS=BLOCK_NS,
    )
    return g