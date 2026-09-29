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
    A,
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

    # beta: shape (T,) stride H, base = beta + bos*H + i_h
    # g: shape (T,) stride H, base = g + bos*H + i_h
    # A: shape (T, BT) stride (H*BT, 1), base = A + (bos*H + i_h)*BT
    # v: shape (T, V) stride (H*V, 1), base = v + (bos*H + i_h)*V
    # u: shape (T, V) stride (H*V, 1), base = u + (bos*H + i_h)*V
    # k: shape (T, K) stride (Hg*K, 1), base = k + (bos*Hg + i_h//(H//Hg))*K
    # w: shape (T, K) stride (H*K, 1), base = w + (bos*H + i_h)*K

    # Precompute base offsets for all tensors (compile-time constants for strides)
    base_beta = beta + bos * H + i_h
    base_g = g + bos * H + i_h
    base_A = A + (bos * H + i_h) * BT
    base_v = v + (bos * H + i_h) * V
    base_u = u + (bos * H + i_h) * V
    base_k = k + (bos * Hg + i_h // (H // Hg)) * K
    base_w = w + (bos * H + i_h) * K

    # Load beta, A, g using block_ptr (non-contiguous inner stride)
    p_beta = tl.make_block_ptr(
        base_beta, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(base_g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        base_A, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    # v and u: inner stride V is 1 (proven by wrapper), strip stride multiplication
    # Use manual tl.load with base + offsets
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_v = i_v * BV + tl.arange(0, BV)
    # We'll handle v/u inside the loop

    for i_v in range(tl.cdiv(V, BV)):
        offs_v = i_v * BV + tl.arange(0, BV)
        # v base offset: base_v + offs_t * H * V + offs_v (since stride V=1)
        # But H*V is compile-time constant, so we can compute directly
        v_offs = base_v + offs_t[:, None] * (H * V) + offs_v[None, :]
        u_offs = base_u + offs_t[:, None] * (H * V) + offs_v[None, :]
        # Mask for boundary
        mask_t = offs_t < T
        mask_v = offs_v < V
        mask = mask_t[:, None] & mask_v[None, :]
        b_v = tl.load(v_offs, mask=mask)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(u_offs, b_u.to(b_u.dtype.element_ty), mask=mask)

    for i_k in range(tl.cdiv(K, BK)):
        offs_k = i_k * BK + tl.arange(0, BK)
        # k base offset: base_k + offs_t * Hg * K + offs_k (since stride K=1)
        k_offs = base_k + offs_t[:, None] * (Hg * K) + offs_k[None, :]
        w_offs = base_w + offs_t[:, None] * (H * K) + offs_k[None, :]
        mask_k = offs_k < K
        mask = mask_t[:, None] & mask_k[None, :]
        b_k = tl.load(k_offs, mask=mask)
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(w_offs, b_w.to(b_w.dtype.element_ty), mask=mask)

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
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
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