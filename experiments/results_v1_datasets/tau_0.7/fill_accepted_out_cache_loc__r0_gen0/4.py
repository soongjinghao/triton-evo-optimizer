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
    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)

    # 分块两阶段归约:当 size_upper 为 2 的幂且可整除 CHUNK 时,
    # 将一维 tl.sum 拆成 [size_upper // CHUNK, CHUNK] 的二维局部归约。
    CHUNK = 64
    if size_upper > 0 and (size_upper & (size_upper - 1)) == 0 and size_upper % CHUNK == 0:
        masks_2d = tl.view(masks, (size_upper // CHUNK, CHUNK))
        dst = tl.sum(tl.sum(masks_2d, axis=1), axis=0)
    else:
        dst = tl.sum(masks)

    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)