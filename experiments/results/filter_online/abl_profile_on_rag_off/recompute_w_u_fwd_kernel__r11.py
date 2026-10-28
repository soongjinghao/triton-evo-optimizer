import torch
import triton
import triton.language as tl
import functools
from collections.abc import Callable
from typing import Any, Optional

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
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_local = i_t
        T_local = T

    # A is now (B, H, T, BT) layout
    # p_A stride: (H*T*BT, T*BT, BT, 1) but we use block_ptr with (T_local, BT) shape
    # stride for A: (H*T*BT, T*BT, BT, 1) -> for block_ptr we need (T_local, BT) with strides (BT, 1)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T_local, BT), (BT, 1), (i_t_local * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    # beta layout: (B, H, T) -> stride (H*T, T, 1)
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T_local,), (1,), (i_t_local * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    # g layout: (B, H, T) -> stride (H*T, T, 1)
    p_g = tl.make_block_ptr(
        g + bos * H + i_h, (T_local,), (1,), (i_t_local * BT,), (BT,), (0,)
    )
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        # v layout: (B, H, T, V) -> stride (H*T*V, T*V, V, 1)
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T_local, V),
            (V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)

        # u layout: (B, H, T, V) -> stride (H*T*V, T*V, V, 1)
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T_local, V),
            (V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    head_group_idx = i_h // (H // Hg)
    for i_k in range(tl.cdiv(K, BK)):
        # k layout: (B, Hg, T, K) -> stride (Hg*T*K, T*K, K, 1)
        p_k = tl.make_block_ptr(
            k + (bos * Hg + head_group_idx) * K,
            (T_local, K),
            (K, 1),
            (i_t_local * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)

        # w layout: (B, H, T, K) -> stride (H*T*K, T*K, K, 1)
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T_local, K),
            (K, 1),
            (i_t_local * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    # Transpose A from (B, T, H, BT) to (B, H, T, BT)
    A_transposed = A.permute(0, 2, 1, 3).contiguous()

    # Transpose beta from (B, T, H) to (B, H, T)
    beta_transposed = beta.permute(0, 2, 1).contiguous()

    # Transpose g_cumsum from (B, T, H) to (B, H, T)
    g_transposed = g_cumsum.permute(0, 2, 1).contiguous()

    # Transpose v from (B, T, H, V) to (B, H, T, V)
    v_transposed = v.permute(0, 2, 1, 3).contiguous()

    # Transpose k from (B, T, Hg, K) to (B, Hg, T, K)
    k_transposed = k.permute(0, 2, 1, 3).contiguous()

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 64
    BV = 64

    # Output u: (B, H, T, V) layout
    u = torch.empty(B, H, T, V, dtype=v.dtype, device=v.device)
    # Output w: (B, H, T, K) layout
    w = torch.empty(B, H, T, K, dtype=k.dtype, device=k.device)

    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k_transposed,
        v=v_transposed,
        beta=beta_transposed,
        w=w,
        u=u,
        A=A_transposed,
        g=g_transposed,
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

    # Transpose outputs back to original layout
    # w: (B, H, T, K) -> (B, T, H, K)
    w_out = w.permute(0, 2, 1, 3).contiguous()
    # u: (B, H, T, V) -> (B, T, H, V)
    u_out = u.permute(0, 2, 1, 3).contiguous()

    return w_out, u_out