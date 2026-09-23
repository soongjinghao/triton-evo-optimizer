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
    pid = tl.program_id(axis=0)

    BLOCK: tl.constexpr = 64
    n_blocks = (size_upper + BLOCK - 1) // BLOCK
    padded_size = n_blocks * BLOCK

    offset = tl.arange(0, padded_size)
    mask = offset < size_upper

    acc_vec = tl.load(accept_index + offset, mask=mask, other=-1)
    cond = (acc_vec != -1).to(tl.int32)

    masked_cond = tl.where(offset < pid, cond, 0)
    masked_cond_2d = tl.view(masked_cond, (n_blocks, BLOCK))
    block_sum = tl.sum(masked_cond_2d, axis=1)
    dst = tl.sum(block_sum)

    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)