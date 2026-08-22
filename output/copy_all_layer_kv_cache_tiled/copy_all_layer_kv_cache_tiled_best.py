import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1, 'num_stages': 1}),
        triton.Config({'num_warps': 2, 'num_stages': 1}),
        triton.Config({'num_warps': 4, 'num_stages': 1}),
        triton.Config({'num_warps': 8, 'num_stages': 1}),
        triton.Config({'num_warps': 1, 'num_stages': 2}),
        triton.Config({'num_warps': 2, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 1, 'num_stages': 3}),
        triton.Config({'num_warps': 2, 'num_stages': 3}),
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 1, 'num_stages': 4}),
        triton.Config({'num_warps': 2, 'num_stages': 4}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
        triton.Config({'num_warps': 8, 'num_stages': 4}),
    ],
    key=['num_locs', 'BYTES_PER_TILE'],
)
@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))
    loc = tl.arange(0, num_locs_upper)
    loc_mask = loc < num_locs
    src = tl.load(src_loc_ptr + loc, mask=loc_mask, other=0)
    tgt = tl.load(tgt_loc_ptr + loc, mask=loc_mask, other=0)
    src_base = src * stride
    tgt_base = tgt * stride
    INNER: tl.constexpr = 64
    tile_base = tid * BYTES_PER_TILE
    num_steps: tl.constexpr = tl.cdiv(BYTES_PER_TILE, INNER)
    rel = tl.arange(0, INNER)
    byte_off = tile_base + rel
    byte_mask = (rel < BYTES_PER_TILE) & (byte_off < stride)
    mask_curr = loc_mask[:, None] & byte_mask[None, :]
    src_ptr = base_ptr + src_base[:, None] + byte_off[None, :]
    tgt_ptr = base_ptr + tgt_base[:, None] + byte_off[None, :]
    vals = tl.load(src_ptr, mask=mask_curr, other=0)
    if num_steps > 1:
        for step in range(1, num_steps):
            next_start = step * INNER
            rel_next = next_start + tl.arange(0, INNER)
            byte_off_next = tile_base + rel_next
            byte_mask_next = (rel_next < BYTES_PER_TILE) & (byte_off_next < stride)
            mask_next = loc_mask[:, None] & byte_mask_next[None, :]
            src_ptr_next = base_ptr + src_base[:, None] + byte_off_next[None, :]
            tgt_ptr_next = base_ptr + tgt_base[:, None] + byte_off_next[None, :]
            next_vals = tl.load(src_ptr_next, mask=mask_next, other=0)
            tl.store(tgt_ptr, vals, mask=mask_curr)
            vals = next_vals
            mask_curr = mask_next
            tgt_ptr = tgt_ptr_next
            src_ptr = src_ptr_next
    tl.store(tgt_ptr, vals, mask=mask_curr)