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
    start_bos,
    start_eos,
    start_i_t,
    start_i_n,
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
    task_idx = pid
    if task_idx >= NT_total:
        return
    i_n = 0
    i_t = 0
    bos = 0
    eos = 0
    T_local = 0
    i_b = 0
    i_h = 0
    if IS_VARLEN:
        i_n = tl.load(start_i_n + task_idx).to(tl.int32)
        i_t = tl.load(start_i_t + task_idx).to(tl.int32)
        bos = tl.load(start_bos + task_idx).to(tl.int32)
        eos = tl.load(start_eos + task_idx).to(tl.int32)
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
    grid = (NT_total,)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    start_bos = None
    start_eos = None
    start_i_t = None
    start_i_n = None
    if cu_seqlens is not None:
        start_bos = torch.empty(NT_total, dtype=torch.int32, device=g.device)
        start_eos = torch.empty(NT_total, dtype=torch.int32, device=g.device)
        start_i_t = torch.empty(NT_total, dtype=torch.int32, device=g.device)
        start_i_n = torch.empty(NT_total, dtype=torch.int32, device=g.device)
        for task_idx in range(NT_total):
            i_n = chunk_indices[task_idx, 0].item()
            i_t = chunk_indices[task_idx, 1].item()
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            start_bos[task_idx] = bos
            start_eos[task_idx] = eos
            start_i_t[task_idx] = i_t
            start_i_n[task_idx] = i_n
    chunk_local_cumsum_scalar_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        start_bos,
        start_eos,
        start_i_t,
        start_i_n,
        NT_total=NT_total,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        BLOCK_BH=1,
        NT_per_seq=NT_per_seq,
    )
    return g