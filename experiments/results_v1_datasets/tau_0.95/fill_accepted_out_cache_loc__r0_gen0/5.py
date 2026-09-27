import triton
import triton.language as tl

@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    BLOCK = 64
    num_blocks = (size_upper + BLOCK - 1) // BLOCK
    pid = tl.program_id(axis=0)

    if pid < num_blocks:
        block_start = pid * BLOCK

        offset = tl.arange(0, size_upper)
        acc_vec = tl.load(accept_index + offset)
        cond = (acc_vec != -1).to(tl.int32)
        base = tl.sum(tl.where(offset < block_start, cond, 0))

        lane = tl.arange(0, BLOCK)
        idx = block_start + lane
        valid = idx < size_upper
        block_acc = tl.load(accept_index + idx, mask=valid, other=-1)
        block_cond = (block_acc != -1).to(tl.int32)

        cumsum = tl.cumsum(block_cond, axis=0)
        dst_vec = base + cumsum - block_cond

        write_mask = valid & (block_acc > -1)
        vals = tl.load(out_cache_loc + block_acc, mask=write_mask, other=0)
        tl.store(accepted_out_cache_loc + dst_vec, vals, mask=write_mask)