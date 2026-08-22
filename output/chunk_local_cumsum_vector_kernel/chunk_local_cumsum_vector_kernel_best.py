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
    BLOCK_BH: tl.constexpr,
    BLOCK_NT: tl.constexpr,
    BLOCK_NS: tl.constexpr,
):
    pid = tl.program_id(0)
    o_i = tl.arange(0, BT)
    mask_reverse = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    mask_forward = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    m_s = tl.where(REVERSE, mask_reverse, mask_forward)
    total_s_tiles = tl.cdiv(S, BS)
    if IS_VARLEN:
        total_nt = 0
        cu_seqlens_vec = tl.load(cu_seqlens + tl.arange(0, B + 1)).to(tl.int32)
    else:
        total_nt = tl.cdiv(T, BT)
    total_bh = B * H
    s_tile_groups = tl.cdiv(total_s_tiles, BLOCK_NS)
    nt_groups = tl.cdiv(total_nt, BLOCK_NT) if not IS_VARLEN else 1
    bh_groups = tl.cdiv(total_bh, BLOCK_BH)
    total_programs = s_tile_groups * nt_groups * bh_groups
    if pid < total_programs:
        group_idx = pid
        group_s = group_idx % s_tile_groups
        group_idx = group_idx // s_tile_groups
        group_nt = group_idx % nt_groups
        group_bh = group_idx // nt_groups
        bh_start = group_bh * BLOCK_BH
        bh_end = min(bh_start + BLOCK_BH, total_bh)
        nt_start = group_nt * BLOCK_NT
        nt_end = min(nt_start + BLOCK_NT, total_nt) if not IS_VARLEN else BLOCK_NT
        s_start = group_s * BLOCK_NS
        s_end = min(s_start + BLOCK_NS, total_s_tiles)
        if IS_VARLEN:
            offs_nt = nt_start + tl.arange(0, BLOCK_NT)
            i_n_vec = tl.load(chunk_indices + offs_nt * 2).to(tl.int32)
            i_t_local_vec = tl.load(chunk_indices + offs_nt * 2 + 1).to(tl.int32)
        for bh_offset in range(BLOCK_BH):
            bh_idx = bh_start + bh_offset
            if bh_idx < bh_end:
                i_b, i_h = bh_idx // H, bh_idx % H
                for nt_offset in range(BLOCK_NT):
                    nt_idx = nt_start + nt_offset
                    if not IS_VARLEN:
                        if nt_idx < nt_end:
                            bos = i_b * T
                            T_curr = T
                            i_t = nt_idx
                            chunk_valid = True
                            for s_offset in range(BLOCK_NS):
                                s_idx = s_start + s_offset
                                if s_idx < s_end:
                                    i_s = s_idx
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
                                    if i_t * BT < T_curr and i_s * BS < S:
                                        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
                                        b_o = tl.dot(m_s, b_s, allow_tf32=False)
                                        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
                    else:
                        if nt_idx >= 0:
                            chunk_valid_idx = nt_idx >= 0
                            if chunk_valid_idx:
                                i_n = i_n_vec[nt_offset]
                                i_t_local = i_t_local_vec[nt_offset]
                                bos = cu_seqlens_vec[i_n]
                                eos = cu_seqlens_vec[i_n + 1]
                                T_curr = eos - bos
                                i_t = i_t_local
                                if i_t * BT < T_curr:
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
    NT = triton.cdiv(T, BT) if cu_seqlens is None else (len(chunk_indices) if chunk_indices is not None else 0)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    BLOCK_BH = 4
    BLOCK_NT = 2
    BLOCK_NS = 2
    BS_max = max(BS_LIST)
    total_s_tiles = triton.cdiv(S, BS_max)
    total_nt = NT
    total_bh = B * H
    s_tile_groups = triton.cdiv(total_s_tiles, BLOCK_NS)
    nt_groups = triton.cdiv(total_nt, BLOCK_NT) if total_nt > 0 else 1
    bh_groups = triton.cdiv(total_bh, BLOCK_BH)
    grid_size = min(s_tile_groups * nt_groups * bh_groups, 40)
    grid_size = max(grid_size, 1)
    chunk_local_cumsum_vector_kernel[(grid_size,)](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        BLOCK_BH=BLOCK_BH,
        BLOCK_NT=BLOCK_NT,
        BLOCK_NS=BLOCK_NS,
    )
    return g