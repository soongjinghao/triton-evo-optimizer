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

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

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

@tensor_cache
def prepare_chunk_meta(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.Tensor:
    chunk_indices, chunk_bounds = prepare_chunk_indices(cu_seqlens, chunk_size)
    bos = chunk_bounds[:, 0].to(torch.int32)
    eos = chunk_bounds[:, 1].to(torch.int32)
    i_t_local = chunk_indices[:, 1].to(torch.int32)
    return torch.stack([bos, eos, i_t_local], dim=1).contiguous()

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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        bos = tl.load(chunk_meta + i_t * 3 + 0).to(tl.int32)
        eos = tl.load(chunk_meta + i_t * 3 + 1).to(tl.int32)
        i_t_local = tl.load(chunk_meta + i_t * 3 + 2).to(tl.int32)
        T_local = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_local = i_t
        T_local = T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T_local,), (H,), (i_t_local * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h), (T_local,), (H,), (i_t_local * BT,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T_local, BT), (H * BT, 1), (i_t_local * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    head_group_idx = i_h // (H // Hg)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + head_group_idx) * K,
            (T_local, K),
            (Hg * K, 1),
            (i_t_local * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T_local, K),
            (H * K, 1),
            (i_t_local * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

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
def recompute_w_u_fwd_kernel_3d(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_meta,
    group_idx,
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
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)
    if IS_VARLEN:
        bos = tl.load(chunk_meta + i_t * 3 + 0).to(tl.int32)
        eos = tl.load(chunk_meta + i_t * 3 + 1).to(tl.int32)
        i_t_local = tl.load(chunk_meta + i_t * 3 + 2).to(tl.int32)
        T_local = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_local = i_t
        T_local = T
    head_group_idx = tl.load(group_idx + i_h)
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T_local,), (H,), (i_t_local * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h), (T_local,), (H,), (i_t_local * BT,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T_local, BT), (H * BT, 1), (i_t_local * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t_local * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + head_group_idx) * K,
            (T_local, K),
            (Hg * K, 1),
            (i_t_local * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T_local, K),
            (H * K, 1),
            (i_t_local * BT, i_k * BK),
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
        chunk_meta = prepare_chunk_meta(cu_seqlens, BT)
        NT = len(chunk_meta)
    else:
        chunk_meta = None
        NT = triton.cdiv(T, BT)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    if H % Hg == 0:
        group_idx = torch.arange(H, dtype=torch.int32, device=k.device) // (H // Hg)
        recompute_w_u_fwd_kernel_3d[(NT, H, B)](
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_meta=chunk_meta,
            group_idx=group_idx,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        )
    else:
        recompute_w_u_fwd_kernel[(NT, B * H)](
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