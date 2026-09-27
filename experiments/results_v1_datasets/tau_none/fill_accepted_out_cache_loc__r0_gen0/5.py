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

    dst = tl.zeros([1], dtype=tl.int32)

    for t in range(size_upper // TILE):
        off = t * TILE + tl.arange(0, TILE)
        mask = off < pid
        x = tl.load(accept_index + off, mask=mask, other=-1)
        dst += tl.sum((x != -1).to(tl.int32))

    tail_start = TILE * (size_upper // TILE)
    tail_size = size_upper - tail_start
    if tail_size > 0:
        off = tail_start + tl.arange(0, tail_size)
        mask = off < pid
        x = tl.load(accept_index + off, mask=mask, other=-1)
        dst += tl.sum((x != -1).to(tl.int32))

    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)