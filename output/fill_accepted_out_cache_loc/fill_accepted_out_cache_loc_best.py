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

    # Load the entire accept_index vector (each program loads the same full vector)
    acc_vec = tl.load(accept_index + tl.arange(0, size_upper))

    # Compute cumulative sum of valid entries (those != -1)
    cond = (acc_vec != -1).to(tl.int32)
    cumsum = tl.cumsum(cond)

    # Shift cumsum to obtain prefix sum: cumsum_shifted[pid] = cumsum[pid-1] (with 0 for pid=0)
    cumsum_shifted = tl.cat([tl.zeros([1], dtype=tl.int32), cumsum], axis=0)
    dst = cumsum_shifted[pid]  # number of valid entries before pid

    # Load the current entry and conditionally store
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)