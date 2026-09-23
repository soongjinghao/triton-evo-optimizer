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
    if pid == 0:
        BLOCK_SIZE: tl.constexpr = 128
        offs = tl.arange(0, BLOCK_SIZE)
        prefix = 0
        for start in range(0, size_upper, BLOCK_SIZE):
            mask = start + offs < size_upper
            acc_vec = tl.load(accept_index + start + offs, mask=mask, other=-1)
            valid_mask = acc_vec != -1
            valid = valid_mask.to(tl.int32)
            cum = tl.cumsum(valid, axis=0)
            dst = prefix + cum - valid
            value = tl.load(out_cache_loc + acc_vec, mask=mask & valid_mask, other=0)
            tl.store(accepted_out_cache_loc + dst, value, mask=mask & valid_mask)
            prefix += tl.sum(valid, axis=0)