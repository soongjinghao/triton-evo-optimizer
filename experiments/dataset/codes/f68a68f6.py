import torch
import triton
import triton.language as tl

@triton.jit
def _precompute_metadata_kernel(
    data_ptrs,
    strides,
    cum_lengths,
    base_ptrs,
    num_bids: tl.constexpr,
):
    bid = tl.program_id(0)
    if bid < num_bids:
        stride = tl.load(strides + bid)
        base_ptr = tl.load(data_ptrs + bid)
        base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))
        tl.store(cum_lengths + bid, stride)
        tl.store(base_ptrs + bid, base_ptr)

@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
    cum_lengths,
    base_ptrs,
    num_bids: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    stride = tl.load(cum_lengths + bid)
    base_ptr = tl.load(base_ptrs + bid)
    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs
    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)
    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]
    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)

def copy_all_layer_kv_cache_tiled_wrapper(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    num_bids = data_ptrs.shape[0]
    cum_lengths = torch.empty(num_bids, dtype=strides.dtype, device=strides.device)
    base_ptrs = torch.empty(num_bids, dtype=torch.int64, device=strides.device)
    grid_pre = (num_bids,)
    _precompute_metadata_kernel[grid_pre](
        data_ptrs, strides, cum_lengths, base_ptrs, num_bids
    )
    grid = (num_bids, triton.cdiv(num_locs_upper * BYTES_PER_TILE, BYTES_PER_TILE))
    copy_all_layer_kv_cache_tiled[grid](
        data_ptrs,
        strides,
        tgt_loc_ptr,
        src_loc_ptr,
        num_locs,
        num_locs_upper,
        BYTES_PER_TILE,
        cum_lengths,
        base_ptrs,
        num_bids,
    )