import torch
import triton
import triton.language as tl

@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)
    
    # Prefetch accept_index data for next iteration
    prefetch_offset = offset + size_upper
    tl.prefetch(accept_index + prefetch_offset)
    
    masks = (tl.load(accept_index + offset, mask=offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    
    # Prefetch out_cache_loc data for potential future load
    src = tl.load(accept_index + pid)
    if src > -1:
        tl.prefetch(out_cache_loc + src)
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)