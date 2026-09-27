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
    TILE = 128
    src = tl.load(accept_index + pid)
    if src > -1:
        counts = tl.zeros((), dtype=tl.int32)
        for start in range(0, size_upper, TILE):
            offsets = start + tl.arange(0, TILE)
            mask = (offsets < pid) & (offsets < size_upper)
            x = tl.load(accept_index + offsets, mask=mask, other=-1)
            counts += tl.sum((x != -1).to(tl.int32))
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + counts, value)