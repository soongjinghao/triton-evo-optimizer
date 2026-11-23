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
    loc_idx = tl.program_id(2)
    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))
    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)
    src = tl.load(src_loc_ptr + loc_idx)
    tgt = tl.load(tgt_loc_ptr + loc_idx)
    src_ptr = base_ptr + src * stride + byte_off
    tgt_ptr = base_ptr + tgt * stride + byte_off
    mask = mask_byte
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)