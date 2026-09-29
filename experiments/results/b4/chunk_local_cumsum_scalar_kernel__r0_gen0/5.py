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

@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "REVERSE"],
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
    BLOCK_BH: tl.constexpr,
    NT_per_seq: tl.constexpr,
):
    pid = tl.program_id(0)
    start_task = pid * BLOCK_BH
    end_task = min(start_task + BLOCK_BH, NT_total)
    for task_idx in range(start_task, end_task):
        i_n = tl.load(chunk_indices + task_idx * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + task_idx * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        i_h = task_idx % H
        offset = bos * H + i_h * T_local + i_t * BT
        b_s = tl.load(s + offset + tl.arange(0, BT), mask=tl.arange(0, BT) < T_local - i_t * BT).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        if REVERSE:
            b_z = tl.sum(b_s, axis=0)
            b_o = -b_o + b_z[None] + b_s
        tl.store(o + offset + tl.arange(0, BT), b_o.to(s.dtype.element_ty), mask=tl.arange(0, BT) < T_local - i_t * BT)

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
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    else:
        cu_seqlens_fake = torch.arange(0, B + 1, device=g.device, dtype=torch.long) * T
        chunk_indices = prepare_chunk_indices(cu_seqlens_fake, BT)
        cu_seqlens = cu_seqlens_fake
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
        REVERSE=reverse,
        BLOCK_BH=BLOCK_BH,
        NT_per_seq=triton.cdiv(T, BT),
    )
    return g