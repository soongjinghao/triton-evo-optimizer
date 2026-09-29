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
    offset = tl.arange(0, size_upper)
    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)

def fill_accepted_out_cache_loc_wrapper(accept_index, out_cache_loc, accepted_out_cache_loc):
    size_upper = accept_index.shape[0]
    grid = (size_upper,)
    fill_accepted_out_cache_loc[grid](
        accept_index, out_cache_loc, accepted_out_cache_loc, size_upper,
        num_warps=4, num_stages=2
    )
    return accepted_out_cache_loc