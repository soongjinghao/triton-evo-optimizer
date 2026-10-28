import warnings
import torch
import triton
import triton.language as tl
BS_LIST = [32, 64]
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
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    NT_total: tl.constexpr,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    NT_per_seq: tl.constexpr,
):
    pid = tl.program_id(0)
    start_task = pid * BLOCK_BH
    end_task = min(start_task + BLOCK_BH, NT_total)
    for task_idx in range(start_task, end_task):
        i_n = 0
        i_t = 0
        bos = 0
        eos = 0
        T_local = 0
        i_b = 0
        i_h = 0
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + task_idx * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + task_idx * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_local = eos - bos
        else:
            i_b = task_idx // (NT_per_seq * H)
            remaining = task_idx % (NT_per_seq * H)
            i_h = remaining % H
            i_t = remaining // H
            bos = i_b * T
            eos = i_b * T + T
            T_local = T
        if HEAD_FIRST:
            if H <= 8:
                offset_base = bos * H
                p_s0 = tl.make_block_ptr(
                    s + offset_base, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                )
                b_s0 = tl.load(p_s0, boundary_check=(0,)).to(tl.float32)
                b_o0 = tl.cumsum(b_s0, axis=0)
                if REVERSE:
                    b_z0 = tl.sum(b_s0, axis=0)
                    b_o0 = -b_o0 + b_z0[None] + b_s0
                p_o0 = tl.make_block_ptr(
                    o + offset_base, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                )
                tl.store(p_o0, b_o0.to(p_o0.dtype.element_ty), boundary_check=(0,))
                if H >= 2:
                    offset_base1 = bos * H + 1 * T_local
                    p_s1 = tl.make_block_ptr(
                        s + offset_base1, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s1 = tl.load(p_s1, boundary_check=(0,)).to(tl.float32)
                    b_o1 = tl.cumsum(b_s1, axis=0)
                    if REVERSE:
                        b_z1 = tl.sum(b_s1, axis=0)
                        b_o1 = -b_o1 + b_z1[None] + b_s1
                    p_o1 = tl.make_block_ptr(
                        o + offset_base1, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o1, b_o1.to(p_o1.dtype.element_ty), boundary_check=(0,))
                if H >= 3:
                    offset_base2 = bos * H + 2 * T_local
                    p_s2 = tl.make_block_ptr(
                        s + offset_base2, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s2 = tl.load(p_s2, boundary_check=(0,)).to(tl.float32)
                    b_o2 = tl.cumsum(b_s2, axis=0)
                    if REVERSE:
                        b_z2 = tl.sum(b_s2, axis=0)
                        b_o2 = -b_o2 + b_z2[None] + b_s2
                    p_o2 = tl.make_block_ptr(
                        o + offset_base2, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o2, b_o2.to(p_o2.dtype.element_ty), boundary_check=(0,))
                if H >= 4:
                    offset_base3 = bos * H + 3 * T_local
                    p_s3 = tl.make_block_ptr(
                        s + offset_base3, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s3 = tl.load(p_s3, boundary_check=(0,)).to(tl.float32)
                    b_o3 = tl.cumsum(b_s3, axis=0)
                    if REVERSE:
                        b_z3 = tl.sum(b_s3, axis=0)
                        b_o3 = -b_o3 + b_z3[None] + b_s3
                    p_o3 = tl.make_block_ptr(
                        o + offset_base3, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o3, b_o3.to(p_o3.dtype.element_ty), boundary_check=(0,))
                if H >= 5:
                    offset_base4 = bos * H + 4 * T_local
                    p_s4 = tl.make_block_ptr(
                        s + offset_base4, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s4 = tl.load(p_s4, boundary_check=(0,)).to(tl.float32)
                    b_o4 = tl.cumsum(b_s4, axis=0)
                    if REVERSE:
                        b_z4 = tl.sum(b_s4, axis=0)
                        b_o4 = -b_o4 + b_z4[None] + b_s4
                    p_o4 = tl.make_block_ptr(
                        o + offset_base4, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o4, b_o4.to(p_o4.dtype.element_ty), boundary_check=(0,))
                if H >= 6:
                    offset_base5 = bos * H + 5 * T_local
                    p_s5 = tl.make_block_ptr(
                        s + offset_base5, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s5 = tl.load(p_s5, boundary_check=(0,)).to(tl.float32)
                    b_o5 = tl.cumsum(b_s5, axis=0)
                    if REVERSE:
                        b_z5 = tl.sum(b_s5, axis=0)
                        b_o5 = -b_o5 + b_z5[None] + b_s5
                    p_o5 = tl.make_block_ptr(
                        o + offset_base5, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o5, b_o5.to(p_o5.dtype.element_ty), boundary_check=(0,))
                if H >= 7:
                    offset_base6 = bos * H + 6 * T_local
                    p_s6 = tl.make_block_ptr(
                        s + offset_base6, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s6 = tl.load(p_s6, boundary_check=(0,)).to(tl.float32)
                    b_o6 = tl.cumsum(b_s6, axis=0)
                    if REVERSE:
                        b_z6 = tl.sum(b_s6, axis=0)
                        b_o6 = -b_o6 + b_z6[None] + b_s6
                    p_o6 = tl.make_block_ptr(
                        o + offset_base6, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o6, b_o6.to(p_o6.dtype.element_ty), boundary_check=(0,))
                if H >= 8:
                    offset_base7 = bos * H + 7 * T_local
                    p_s7 = tl.make_block_ptr(
                        s + offset_base7, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    b_s7 = tl.load(p_s7, boundary_check=(0,)).to(tl.float32)
                    b_o7 = tl.cumsum(b_s7, axis=0)
                    if REVERSE:
                        b_z7 = tl.sum(b_s7, axis=0)
                        b_o7 = -b_o7 + b_z7[None] + b_s7
                    p_o7 = tl.make_block_ptr(
                        o + offset_base7, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                    )
                    tl.store(p_o7, b_o7.to(p_o7.dtype.element_ty), boundary_check=(0,))
            else:
                for h_idx in tl.range(0, H):
                    i_h = h_idx
                    if IS_VARLEN:
                        offset_s = bos * H + i_h * T_local
                        offset_o = bos * H + i_h * T_local
                        p_s = tl.make_block_ptr(
                            s + offset_s, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                        )
                        p_o = tl.make_block_ptr(
                            o + offset_o, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                        )
                    else:
                        offset_s = bos * H + i_h * T_local
                        offset_o = bos * H + i_h * T_local
                        p_s = tl.make_block_ptr(
                            s + offset_s, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                        )
                        p_o = tl.make_block_ptr(
                            o + offset_o, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                        )
                    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
                    b_o = tl.cumsum(b_s, axis=0)
                    if REVERSE:
                        b_z = tl.sum(b_s, axis=0)
                        b_o = -b_o + b_z[None] + b_s
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        else:
            if IS_VARLEN:
                i_h = task_idx % H
                offset_s = bos * H + i_h
                offset_o = bos * H + i_h
                p_s = tl.make_block_ptr(s + offset_s, (T_local,), (H,), (i_t * BT,), (BT,), (0,))
                p_o = tl.make_block_ptr(o + offset_o, (T_local,), (H,), (i_t * BT,), (BT,), (0,))
            else:
                offset_s = bos * H + i_h
                offset_o = bos * H + i_h
                p_s = tl.make_block_ptr(s + offset_s, (T_local,), (H,), (i_t * BT,), (BT,), (0,))
                p_o = tl.make_block_ptr(o + offset_o, (T_local,), (H,), (i_t * BT,), (BT,), (0,))
            b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=0)
            if REVERSE:
                b_z = tl.sum(b_s, axis=0)
                b_o = -b_o + b_z[None] + b_s
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT_per_seq = triton.cdiv(T, BT)
    if cu_seqlens is None:
        NT_total = B * H * NT_per_seq
    else:
        if head_first:
            NT_total = len(chunk_indices) * H
        else:
            NT_total = len(chunk_indices) * H
    MAX_GRID_SIZE = 40
    BLOCK_BH = triton.cdiv(NT_total, MAX_GRID_SIZE)
    grid_size = min(MAX_GRID_SIZE, triton.cdiv(NT_total, BLOCK_BH))
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (grid_size,)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        NT_total=NT_total,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        BLOCK_BH=BLOCK_BH,
        NT_per_seq=NT_per_seq,
    )
    return g