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
    i_t_pair, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        i_n0, i_t0 = (
            tl.load(chunk_indices + (i_t_pair * 2) * 2).to(tl.int32),
            tl.load(chunk_indices + (i_t_pair * 2) * 2 + 1).to(tl.int32),
        )
        bos0, eos0 = (
            tl.load(cu_seqlens + i_n0).to(tl.int32),
            tl.load(cu_seqlens + i_n0 + 1).to(tl.int32),
        )
        T0 = eos0 - bos0
        
        has_second = (i_t_pair * 2 + 1) < NT
        if has_second:
            i_n1, i_t1 = (
                tl.load(chunk_indices + (i_t_pair * 2 + 1) * 2).to(tl.int32),
                tl.load(chunk_indices + (i_t_pair * 2 + 1) * 2 + 1).to(tl.int32),
            )
            bos1, eos1 = (
                tl.load(cu_seqlens + i_n1).to(tl.int32),
                tl.load(cu_seqlens + i_n1 + 1).to(tl.int32),
            )
            T1 = eos1 - bos1
        else:
            bos1, eos1 = 0, 0
            T1 = 0
    else:
        bos, eos = i_b * T, i_b * T + T
        bos0, eos0 = bos, eos
        T0 = T
        has_second = (i_t_pair * 2 + 1) < NT
        if has_second:
            bos1, eos1 = bos, eos
            T1 = T
        else:
            bos1, eos1 = 0, 0
            T1 = 0
    
    # Load beta for both chunks: (2*BT,)
    p_beta = tl.make_block_ptr(
        beta + bos0 * H + i_h, (T0,), (H,), (i_t0 * BT,), (BT,), (0,)
    )
    b_beta0 = tl.load(p_beta, boundary_check=(0,))
    
    if has_second:
        p_beta1 = tl.make_block_ptr(
            beta + bos1 * H + i_h, (T1,), (H,), (i_t1 * BT,), (BT,), (0,)
        )
        b_beta1 = tl.load(p_beta1, boundary_check=(0,))
    else:
        b_beta1 = tl.zeros((BT,), dtype=b_beta0.dtype)
    
    # Load A for both chunks: (2*BT, BT)
    p_A0 = tl.make_block_ptr(
        A + (bos0 * H + i_h) * BT, (T0, BT), (H * BT, 1), (i_t0 * BT, 0), (BT, BT), (1, 0)
    )
    b_A0 = tl.load(p_A0, boundary_check=(0, 1))
    
    if has_second:
        p_A1 = tl.make_block_ptr(
            A + (bos1 * H + i_h) * BT, (T1, BT), (H * BT, 1), (i_t1 * BT, 0), (BT, BT), (1, 0)
        )
        b_A1 = tl.load(p_A1, boundary_check=(0, 1))
    else:
        b_A1 = tl.zeros((BT, BT), dtype=b_A0.dtype)
    
    # Load g for both chunks
    p_g0 = tl.make_block_ptr(g + (bos0 * H + i_h), (T0,), (H,), (i_t0 * BT,), (BT,), (0,))
    b_g0 = tl.exp(tl.load(p_g0, boundary_check=(0,)))
    
    if has_second:
        p_g1 = tl.make_block_ptr(g + (bos1 * H + i_h), (T1,), (H,), (i_t1 * BT,), (BT,), (0,))
        b_g1 = tl.exp(tl.load(p_g1, boundary_check=(0,)))
    else:
        b_g1 = tl.zeros((BT,), dtype=b_g0.dtype)
    
    # Process V dimension for both chunks
    for i_v in range(tl.cdiv(V, BV)):
        # Load v for chunk 0
        p_v0 = tl.make_block_ptr(
            v + (bos0 * H + i_h) * V,
            (T0, V),
            (H * V, 1),
            (i_t0 * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v0 = tl.load(p_v0, boundary_check=(0, 1))
        b_vb0 = (b_v0 * b_beta0[:, None]).to(b_v0.dtype)
        
        # Load v for chunk 1
        if has_second:
            p_v1 = tl.make_block_ptr(
                v + (bos1 * H + i_h) * V,
                (T1, V),
                (H * V, 1),
                (i_t1 * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_v1 = tl.load(p_v1, boundary_check=(0, 1))
            b_vb1 = (b_v1 * b_beta1[:, None]).to(b_v1.dtype)
        else:
            b_vb1 = tl.zeros((BT, BV), dtype=b_v0.dtype)
        
        # Compute u for chunk 0
        p_u0 = tl.make_block_ptr(
            u + (bos0 * H + i_h) * V,
            (T0, V),
            (H * V, 1),
            (i_t0 * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_u0 = tl.dot(b_A0, b_vb0, allow_tf32=False)
        tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
        
        # Compute u for chunk 1
        if has_second:
            p_u1 = tl.make_block_ptr(
                u + (bos1 * H + i_h) * V,
                (T1, V),
                (H * V, 1),
                (i_t1 * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_u1 = tl.dot(b_A1, b_vb1, allow_tf32=False)
            tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
    
    # Process K dimension for both chunks
    for i_k in range(tl.cdiv(K, BK)):
        # Load k for chunk 0
        p_k0 = tl.make_block_ptr(
            k + (bos0 * Hg + i_h // (H // Hg)) * K,
            (T0, K),
            (Hg * K, 1),
            (i_t0 * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_kb0 = (b_k0 * b_beta0[:, None] * b_g0[:, None]).to(b_k0.dtype)
        
        # Load k for chunk 1
        if has_second:
            p_k1 = tl.make_block_ptr(
                k + (bos1 * Hg + i_h // (H // Hg)) * K,
                (T1, K),
                (Hg * K, 1),
                (i_t1 * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_kb1 = (b_k1 * b_beta1[:, None] * b_g1[:, None]).to(b_k1.dtype)
        else:
            b_kb1 = tl.zeros((BT, BK), dtype=b_k0.dtype)
        
        # Compute w for chunk 0
        p_w0 = tl.make_block_ptr(
            w + (bos0 * H + i_h) * K,
            (T0, K),
            (H * K, 1),
            (i_t0 * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_w0 = tl.dot(b_A0, b_kb0)
        tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
        
        # Compute w for chunk 1
        if has_second:
            p_w1 = tl.make_block_ptr(
                w + (bos1 * H + i_h) * K,
                (T1, K),
                (H * K, 1),
                (i_t1 * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_w1 = tl.dot(b_A1, b_kb1)
            tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))

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
    
    # Grid consolidation: reduce grid from (NT, B*H) to (NT//2 + NT%2, B*H)
    # When NT < 4, keep original grid to avoid overhead
    if NT < 4:
        grid = (NT, B * H)
    else:
        grid = ((NT + 1) // 2, B * H)
    
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    
    recompute_w_u_fwd_kernel[grid](
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