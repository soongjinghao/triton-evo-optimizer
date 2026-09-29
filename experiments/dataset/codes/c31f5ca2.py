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

def prepare_precomputed_meta(
    cu_seqlens: torch.LongTensor,
    chunk_indices: torch.LongTensor,
    T: int,
    H: int,
    BT: int,
    NT_per_seq: int,
    head_first: bool,
    is_varlen: bool,
) -> torch.Tensor:
    if is_varlen:
        NT_total = len(chunk_indices) * H
    else:
        B = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
        NT_total = B * H * NT_per_seq
    meta = torch.zeros((NT_total, 5), dtype=torch.int32, device=cu_seqlens.device)
    if is_varlen:
        for task_idx in range(len(chunk_indices)):
            i_n = chunk_indices[task_idx, 0].item()
            i_t = chunk_indices[task_idx, 1].item()
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            T_local = eos - bos
            for h_idx in range(H):
                global_idx = task_idx * H + h_idx
                meta[global_idx, 0] = i_n
                meta[global_idx, 1] = i_t
                meta[global_idx, 2] = bos
                meta[global_idx, 3] = eos
                meta[global_idx, 4] = T_local
    else:
        B = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
        for i_b in range(B):
            bos = i_b * T
            eos = i_b * T + T
            T_local = T
            for i_h in range(H):
                for i_t in range(NT_per_seq):
                    global_idx = i_b * H * NT_per_seq + i_h * NT_per_seq + i_t
                    meta[global_idx, 0] = i_b
                    meta[global_idx, 1] = i_t
                    meta[global_idx, 2] = bos
                    meta[global_idx, 3] = eos
                    meta[global_idx, 4] = T_local
    return meta

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
    precomputed_meta,
    meta_stride: tl.constexpr,
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
        meta_base = precomputed_meta + task_idx * meta_stride
        i_n = tl.load(meta_base).to(tl.int32)
        i_t = tl.load(meta_base + 1).to(tl.int32)
        bos = tl.load(meta_base + 2).to(tl.int32)
        eos = tl.load(meta_base + 3).to(tl.int32)
        T_local = tl.load(meta_base + 4).to(tl.int32)
        if HEAD_FIRST:
            for h_idx in tl.range(0, H):
                i_h = h_idx
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
            i_h = task_idx % H
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
    is_varlen = cu_seqlens is not None
    precomputed_meta = prepare_precomputed_meta(
        cu_seqlens if is_varlen else torch.arange(B + 1, device=g.device) * T,
        chunk_indices if is_varlen else torch.zeros((NT_total // H, 2), dtype=torch.int64, device=g.device),
        T,
        H,
        BT,
        NT_per_seq,
        head_first,
        is_varlen,
    )
    meta_stride = 5
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
        precomputed_meta,
        meta_stride=meta_stride,
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