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
) -> tuple[torch.LongTensor, torch.LongTensor]:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(lengths, chunk_size).tolist()
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
    chunk_indices,
    chunk_bounds,
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
    i_tile_pack, i_bh = tl.program_id(0), tl.program_id(1)
    n_v = tl.cdiv(V, BV)
    n_k = tl.cdiv(K, BK)
    n_tiles = n_v + n_k
    i_t = i_tile_pack // n_tiles
    i_tile = i_tile_pack % n_tiles
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(chunk_bounds + i_t * 2).to(tl.int32)
        eos = tl.load(chunk_bounds + i_t * 2 + 1).to(tl.int32)
        T_local = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        i_t_local = i_t
        T_local = T
    bh_offset = bos * H + i_h
    head_group_idx = i_h // (H // Hg)
    hg_offset = bos * Hg + head_group_idx
    t_off = i_t_local * BT
    v_base = v + bh_offset * V
    u_base = u + bh_offset * V
    k_base = k + hg_offset * K
    w_base = w + bh_offset * K
    p_beta = tl.make_block_ptr(
        beta + bh_offset, (T_local,), (H,), (t_off,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + bh_offset, (T_local,), (H,), (t_off,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + bh_offset * BT,
        (T_local, BT),
        (H * BT, 1),
        (t_off, 0),
        (BT, BT),
        (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    if i_tile < n_v:
        i