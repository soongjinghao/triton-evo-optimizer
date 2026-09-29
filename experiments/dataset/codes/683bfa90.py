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
    # Grid consolidation: bid and tid merged into a single program_id(0)
    # Each program handles 4 (bid, tid) pairs via loop unrolling
    pid = tl.program_id(0)
    num_tids = tl.num_programs(0)
    # Decode bid and tid from consolidated pid
    # tid dimension: stride / BYTES_PER_TILE (rounded up)
    stride_tid = tl.cdiv(stride, BYTES_PER_TILE)  # This is dynamic, need to load stride first
    # Actually, we need to precompute num_tids_per_bid or use a different approach
    # Since stride is dynamic per bid, we cannot statically decode bid/tid from a single pid
    # Instead, we keep the original 2D grid but unroll the num_locs loop
    
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    
    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))
    
    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    
    # Unroll num_locs loop: process 4 locations per iteration
    # Use tl.static_range for compile-time unrolling hint
    for i in tl.static_range(0, num_locs_upper, 4):
        loc_idx = i + tl.arange(0, 4)
        mask_loc = loc_idx < num_locs
        
        src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
        tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)
        
        # Batch load: src[:, None] * stride + byte_off[None, :]
        src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
        tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]
        
        mask = mask_loc[:, None] & mask_byte[None, :]
        vals = tl.load(src_ptr, mask=mask)
        tl.store(tgt_ptr, vals, mask=mask)