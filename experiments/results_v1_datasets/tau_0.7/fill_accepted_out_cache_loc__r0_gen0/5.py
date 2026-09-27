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
    NUM_CHUNKS = 4

    if size_upper % NUM_CHUNKS == 0:
        chunk_size = size_upper // NUM_CHUNKS
        offs = tl.arange(0, chunk_size)
        masks = (tl.load(accept_index + offs, offs < pid, other=-1) != -1).to(tl.int64)
        dst = tl.sum(masks)

        for c in tl.static_range(1, NUM_CHUNKS):
            offs = c * chunk_size + tl.arange(0, chunk_size)
            masks = (
                tl.load(accept_index + offs, offs < pid, other=-1) != -1
            ).to(tl.int64)
            dst = dst + tl.sum(masks)
    else:
        offset = tl.arange(0, size_upper)
        masks = (
            tl.load(accept_index + offset, offset < pid, other=-1) != -1
        ).to(tl.int64)
        dst = tl.sum(masks)

    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)