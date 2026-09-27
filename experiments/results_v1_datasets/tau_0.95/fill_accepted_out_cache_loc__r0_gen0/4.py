import torch
import triton
import triton.language as tl
import torch_npu
device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream

@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    TILE: tl.constexpr = 64
    num_tiles = (size_upper + TILE - 1) // TILE
    pid = tl.program_id(axis=0)

    if pid < num_tiles:
        tile_start = pid * TILE
        tile_off = tile_start + tl.arange(0, TILE)

        src_vec = tl.load(accept_index + tile_off, mask=tile_off < size_upper, other=-1)
        cond_vec = (src_vec != -1).to(tl.int32)

        offset = tl.arange(0, size_upper)
        cond_full = (tl.load(accept_index + offset) != -1).to(tl.int32)
        base = tl.sum(tl.where(offset < tile_start, cond_full, 0))

        localsum = tl.cumsum(cond_vec, axis=0)
        dst_vec = base + localsum - cond_vec

        values = tl.load(out_cache_loc + src_vec, mask=src_vec != -1, other=0)
        tl.store(accepted_out_cache_loc + dst_vec, values, mask=src_vec != -1)