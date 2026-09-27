import torch
import triton
import triton.language as tl


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    TILE_H: tl.constexpr = 8
    TILE_W: tl.constexpr = 32
    TILE_AREA: tl.constexpr = TILE_H * TILE_W

    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    kv_len = kv_end - kv_start

    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset

    off_2d = tl.arange(0, TILE_H)[:, None] * TILE_W + tl.arange(0, TILE_W)[None, :]

    num_tiles = tl.cdiv(pool_len, TILE_AREA)
    for tile_id in range(num_tiles):
        offs = tile_id * TILE_AREA + off_2d
        mask = offs < kv_len

        data = tl.load(token_pool + kv_start + offs, mask=mask)
        tl.store(out_cache_ptr + offs, data, mask=mask)


def assign_extend_cache_locs_func(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    grid = (req_pool_indices_tensor.numel(),)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor