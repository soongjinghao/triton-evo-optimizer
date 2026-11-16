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
    # 必须保持真实 wrapper 的 2D launch contract:
    # grid = (num_layers, cdiv(stride, BYTES_PER_TILE))
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    # 同一 program 内一次拿到全部 loc metadata。
    # 这样每个 byte chunk 都会先完成所有 source load，再执行 store，
    # 保持 in-place copy 的 read-before-write 语义。
    loc = tl.arange(0, num_locs_upper)
    loc_mask = loc < num_locs

    src = tl.load(src_loc_ptr + loc, mask=loc_mask, other=0)
    tgt = tl.load(tgt_loc_ptr + loc, mask=loc_mask, other=0)

    src_base = src * stride
    tgt_base = tgt * stride

    # 控制单次二维 tile 面积，避免一次展开 num_locs_upper * BYTES_PER_TILE
    # 造成过高的 UB / register 压力。
    INNER: tl.constexpr = 64
    tile_base = tid * BYTES_PER_TILE

    for start in range(0, BYTES_PER_TILE, INNER):
        rel = start + tl.arange(0, INNER)
        byte_off = tile_base + rel

        # 同时保证：
        # 1) 当前 chunk 不越过本 tile 的 BYTES_PER_TILE；
        # 2) 最后一个 tile 不越过真实 stride。
        byte_mask = (rel < BYTES_PER_TILE) & (byte_off < stride)
        mask = loc_mask[:, None] & byte_mask[None, :]

        src_ptr = base_ptr + src_base[:, None] + byte_off[None, :]
        tgt_ptr = base_ptr + tgt_base[:, None] + byte_off[None, :]

        vals = tl.load(src_ptr, mask=mask, other=0)
        tl.store(tgt_ptr, vals, mask=mask)