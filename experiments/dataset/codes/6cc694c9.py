import torch
import triton
import triton.language as tl
import functools
from collections.abc import Callable
from typing import Any

def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result
    return wrapper

@tensor_cache
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
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A_T,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    # A_T is pre-transposed: shape (BT, T), stride (1, H*BT)
    p_A_T = tl.make_block_ptr(
        A_T + (bos * H + i_h) * BT, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A_T = tl.load(p_A_T, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        # b_A_T is (BT, BT), b_vb is (BT, BV) -> b_u is (BT, BV)
        b_u = tl.dot(b_A_T, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        # b_A_T is (BT, BT), b_kb is (BT, BK) -> b_w is (BT, BK)
        b_w = tl.dot(b_A_T, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    # Pre-transpose A: shape (B, T, H, BT) -> (B, T, H, BT) but with last two dims swapped
    # A_T shape: (B, T, H, BT) with stride (H*BT*T, H*BT, BT, 1) -> we need (BT, T) per (b,h)
    # Actually A is (B, T, H, BT) -> we want per (b,h): (BT, T) contiguous in row-major
    # A_T = A.transpose(-2, -1).contiguous() gives (B, T, BT, H) -> not what we want
    # We need per (b,h): (BT, T) contiguous. A has shape (B, T, H, BT). For each (b,h), A[b,:,h,:] is (T, BT).
    # We want A_T[b,:,h,:] = A[b,:,h,:].T which is (BT, T). We'll create a contiguous tensor.
    A_T = A.transpose(-2, -1).contiguous()  # (B, T, BT, H) -> per (b,h): (BT, T) but layout is (T, BT, H) -> need to reorder
    # Actually simpler: reshape A to (B*T, H, BT), transpose last two dims, then reshape back
    A_2d = A.view(-1, H, BT)  # (B*T, H, BT)
    A_T_2d = A_2d.transpose(-2, -1).contiguous()  # (B*T, BT, H)
    A_T = A_T_2d.view(B, T, BT, H).transpose(-2, -1).contiguous()  # (B, T, H, BT) with (BT, T) per (b,h)
    # Actually we want per (b,h): (BT, T) contiguous. Let's just do:
    # A_T = torch.empty(B, T, H, BT, device=A.device, dtype=A.dtype)
    # for b in range(B):
    #     for h in range(H):
    #         A_T[b, :, h, :] = A[b, :, h, :].T
    # But that's slow. Instead, we can use einops or just permute.
    # A shape: (B, T, H, BT). We want (B, H, BT, T) then view as (B*H, BT, T)
    A_perm = A.permute(0, 2, 3, 1).contiguous()  # (B, H, BT, T)
    A_T = A_perm.view(B * H, BT, T)  # (B*H, BT, T) -> each (b,h) has (BT, T) contiguous
    # Now A_T has shape (B*H, BT, T), stride (BT*T, T, 1) -> per (b,h): (BT, T) row-major
    # We'll pass A_T directly and index with bos*H + i_h
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A_T=A_T,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u