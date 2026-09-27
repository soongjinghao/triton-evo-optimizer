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
    BLOCK_SIZE: tl.constexpr = 128
    
    # Load accept_index[0:pid] in blocks of BLOCK_SIZE, unrolled 4 times
    dst = tl.full([1], 0, tl.int64)
    base_offset = 0
    num_full_blocks = pid // BLOCK_SIZE
    remainder = pid % BLOCK_SIZE
    
    # Process full blocks, unrolled 4 times
    i = 0
    while i < num_full_blocks:
        offset = tl.arange(0, BLOCK_SIZE)
        idx = base_offset + offset
        accept_vals = tl.load(accept_index + idx)
        masks = (accept_vals != -1).to(tl.int64)
        dst += tl.sum(masks)
        base_offset += BLOCK_SIZE
        i += 1
        
        if i < num_full_blocks:
            offset = tl.arange(0, BLOCK_SIZE)
            idx = base_offset + offset
            accept_vals = tl.load(accept_index + idx)
            masks = (accept_vals != -1).to(tl.int64)
            dst += tl.sum(masks)
            base_offset += BLOCK_SIZE
            i += 1
        
        if i < num_full_blocks:
            offset = tl.arange(0, BLOCK_SIZE)
            idx = base_offset + offset
            accept_vals = tl.load(accept_index + idx)
            masks = (accept_vals != -1).to(tl.int64)
            dst += tl.sum(masks)
            base_offset += BLOCK_SIZE
            i += 1
        
        if i < num_full_blocks:
            offset = tl.arange(0, BLOCK_SIZE)
            idx = base_offset + offset
            accept_vals = tl.load(accept_index + idx)
            masks = (accept_vals != -1).to(tl.int64)
            dst += tl.sum(masks)
            base_offset += BLOCK_SIZE
            i += 1
    
    # Process remainder
    if remainder > 0:
        offset = tl.arange(0, BLOCK_SIZE)
        idx = base_offset + offset
        accept_vals = tl.load(accept_index + idx, mask=offset < remainder, other=-1)
        masks = (accept_vals != -1).to(tl.int64)
        dst += tl.sum(masks)
    
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)