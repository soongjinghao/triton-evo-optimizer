import torch
import triton
import triton.language as tl
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

    num_full_iters: tl.constexpr = BYTES_PER_TILE // INNER
    remainder: tl.constexpr = BYTES_PER_TILE % INNER

    # 全覆盖段,省略 rel < BYTES_PER_TILE
    for i in range(num_full_iters):
        start = i * INNER
        rel = start + tl.arange(0, INNER)
        byte_off = tile_base + rel
        byte_mask = byte_off < stride
        mask = loc_mask[:, None] & byte_mask[None, :]
        src_ptr = base_ptr + src_base[:, None] + byte_off[None, :]
        tgt_ptr = base_ptr + tgt_base[:, None] + byte_off[None, :]
        vals = tl.load(src_ptr, mask=mask, other=0)
        tl.store(tgt_ptr, vals, mask=mask)

    # 边界段,恢复完整 mask
    if remainder > 0:
        start = num_full_iters * INNER
        rel = start + tl.arange(0, INNER)
        byte_off = tile_base + rel
        byte_mask = (rel < BYTES_PER_TILE) & (byte_off < stride)
        mask = loc_mask[:, None] & byte_mask[None, :]
        src_ptr = base_ptr + src_base[:, None] + byte_off[None, :]
        tgt_ptr = base_ptr + tgt_base[:, None] + byte_off[None, :]
        vals = tl.load(src_ptr, mask=mask, other=0)
        tl.store(tgt_ptr, vals, mask=mask)