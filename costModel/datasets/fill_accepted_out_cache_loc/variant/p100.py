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
    offset = tl.arange(0, 4 * size_upper)
    masks = (tl.load(accept_index + offset, offset < 4 * pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + 4 * pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)
    src = tl.load(accept_index + 4 * pid + 1)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst + 1, value)
    src = tl.load(accept_index + 4 * pid + 2)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst + 2, value)
    src = tl.load(accept_index + 4 * pid + 3)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst + 3, value)