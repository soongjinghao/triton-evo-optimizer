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
    
    # Strategy: Grid consolidation - each program handles 4 rows
    # Fallback to single row mode when size_upper < 4
    if size_upper < 4:
        offset = tl.arange(0, size_upper)
        masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
        dst = tl.sum(masks)
        src = tl.load(accept_index + pid)
        if src > -1:
            value = tl.load(out_cache_loc + src)
            tl.store(accepted_out_cache_loc + dst, value)
    else:
        # Process 4 rows per program
        base_row = pid * 4
        # Handle tail rows with mask
        row_mask = base_row + tl.arange(0, 4) < size_upper
        
        # Compute dst for each row
        for i in tl.static_range(4):
            row = base_row + i
            if row < size_upper:
                offset = tl.arange(0, size_upper)
                masks = (tl.load(accept_index + offset, offset < row, other=-1) != -1).to(tl.int64)
                dst = tl.sum(masks)
                src = tl.load(accept_index + row)
                if src > -1:
                    value = tl.load(out_cache_loc + src)
                    tl.store(accepted_out_cache_loc + dst, value)