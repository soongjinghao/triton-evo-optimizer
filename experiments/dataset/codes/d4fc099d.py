import torch
import triton
import triton.language as tl
import functools
from collections.abc import Callable
from typing import Any, Optional

def prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]

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
) -> tuple[torch.LongTensor, torch.LongTensor]:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    batch_ids = chunk_indices[:, 0].long()
    bos = cu_seqlens[batch_ids]
    eos = cu_seqlens[batch_ids + 1]
    chunk_bounds = torch.stack([bos, eos], dim=1)
    return chunk_indices, chunk_bounds

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
    chunk_meta,
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
    pid0 = tl.program_id(0)
    i_bh = tl.program_id(1)

    n_u_blocks = tl.cdiv(V, BV)
    n_w_blocks = tl.cdiv(K, BK)
    n_col_blocks = n_u_blocks + n_w_blocks

    i_t = pid0 // n_col_blocks
    col = pid0 % n_col_blocks

    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        p_meta = tl.make_block_ptr(
            chunk_meta + i_t * 3, (3,), (1,), (0,), (3,), (0,)
        )
        meta = tl.load(p_meta, boundary_check=(0,))
        i_t_local = meta[0]
        bos = meta[1]
        eos = meta[2]
        T_local = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_local = i_t
        T_local = T

    head_group_idx = i_h // (H // Hg)
    bh = bos * H + i_h
    bg = bos * Hg + head_group_idx
    t_off = i_t_local * BT

    p_beta = tl.make_block_ptr(
        beta + bh, (T_local,), (H,), (t_off,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + bh * BT, (T_local, BT), (H * BT, 1), (t_off, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    if col < n_u_blocks:
        i_v = col
        v_base = v + bh * V
        u_base = u + bh * V
        p_v = tl.make_block_ptr(
            v_base,
            (T_local, V),
            (H * V, 1),
            (t_off, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u_base,
            (T_local, V),
            (H * V, 1),
            (t_off, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    else:
        i_k = col - n_u_blocks
        k_base = k + bg * K
        w_base = w + bh * K
        p_g = tl.make_block_ptr(
            g + bh, (T_local,), (H,), (t_off,), (BT,), (0,)
        )
        b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
        p_k = tl.make_block_ptr(
            k_base,
            (T_local, K),
            (Hg * K, 1),
            (t_off, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w_base,
            (T_local, K),
            (H * K, 1),
            (t_off, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if cu_seqlens is not None:
        chunk_indices, chunk_bounds = prepare_chunk_indices(cu_seqlens, BT)
        chunk_meta = torch.stack(
            [chunk_indices[:, 1], chunk_bounds[:, 0], chunk_bounds[:, 1]], dim=-1
        ).to(torch.int32)
        NT = len(chunk_indices)
    else:
        chunk_meta = None
        NT = triton.cdiv(T, BT)

    BK = 64
    BV = 64

    n_col_blocks = triton.cdiv(V, BV) + triton.cdiv(K, BK)

    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)

    recompute_w_u_fwd_kernel[(NT * n_col_blocks, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_meta=chunk_meta,
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