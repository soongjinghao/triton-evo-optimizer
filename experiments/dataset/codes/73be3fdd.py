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

def prepare_meta(
    chunk_indices: torch.LongTensor,
    cu_seqlens: torch.LongTensor,
    B: int,
    H: int,
    T: int,
    BT: int,
    head_first: bool,
    is_varlen: bool,
    NT_per_seq: int,
    NT_total: int,
    device: torch.device,
) -> torch.Tensor:
    if is_varlen:
        i_n = chunk_indices[:, 0].to(torch.int32)
        i_t = chunk_indices[:, 1].to(torch.int32)
        bos = cu_seqlens[i_n].to(torch.int32)
        eos = cu_seqlens[i_n + 1].to(torch.int32)
        T_local = eos - bos
        if head_first:
            meta = torch.stack([i_n, i_t, bos, eos, T_local], dim=1).to(device)
            meta = meta.repeat_interleave(H, dim=0)
        else:
            meta = torch.stack([i_n, i_t, bos, eos, T_local], dim=1).to(device)
            meta = meta.repeat_interleave(H, dim=0)
    else:
        total_tasks = B * H * NT_per_seq
        i_b = torch.arange(total_tasks, device=device) // (NT_per_seq * H)
        remaining = torch.arange(total_tasks, device=device) % (NT_per_seq * H)
        i_h = remaining % H
        i_t = remaining // H
        bos = i_b * T
        eos = i_b * T + T
        T_local = torch.full_like(bos, T, dtype=torch.int32)
        meta = torch.stack([i_b, i_t, bos, eos, T_local], dim=1).to(torch.int32)
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
    meta,
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
    end_task = tl.minimum(start_task + BLOCK_BH, NT_total)
    
    if HEAD_FIRST:
        for task_idx in range(start_task, end_task):
            meta_ptr = meta + task_idx * 5
            i_n = tl.load(meta_ptr, mask=task_idx < NT_total).to(tl.int32)
            i_t = tl.load(meta_ptr + 1, mask=task_idx < NT_total).to(tl.int32)
            bos = tl.load(meta_ptr + 2, mask=task_idx < NT_total).to(tl.int32)
            eos = tl.load(meta_ptr + 3, mask=task_idx < NT_total).to(tl.int32)
            T_local = tl.load(meta_ptr + 4, mask=task_idx < NT_total).to(tl.int32)
            
            for h_idx in tl.range(0, H):
                offset_s = bos * H + h_idx * T_local
                offset_o = bos * H + h_idx * T_local
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
        for task_idx in range(start_task, end_task):
            meta_ptr = meta + task_idx * 5
            i_n = tl.load(meta_ptr, mask=task_idx < NT_total).to(tl.int32)
            i_t = tl.load(meta_ptr + 1, mask=task_idx < NT_total).to(tl.int32)
            bos = tl.load(meta_ptr + 2, mask=task_idx < NT_total).to(tl.int32)
            eos = tl.load(meta_ptr + 3, mask=task_idx < NT_total).to(tl.int32)
            T_local = tl.load(meta_ptr + 4, mask=task_idx < NT_total).to(tl.int32)
            
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
    
    meta = prepare_meta(
        chunk_indices, cu_seqlens, B, H, T, BT, head_first,
        cu_seqlens is not None, NT_per_seq, NT_total, g.device
    )
    
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
        meta,
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