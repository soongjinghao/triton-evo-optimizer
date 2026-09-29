import torch
import triton
import triton.language as tl

@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    
    # 外提:一次加载连续的kv_start和kv_end
    kv_start_end = tl.load(start_offset + pid * 2, mask=True)
    kv_start = kv_start_end
    kv_end = tl.load(end_offset + pid * 2 + 1, mask=True)
    
    # 外提:预计算out_offset
    out_offset = tl.load(out_cache_loc + pid, mask=True)
    out_cache_ptr = out_cache_loc + out_offset
    
    # 外提:预计算token_pool基地址
    req_pool_idx = tl.load(req_pool_indices + pid, mask=True)
    token_pool = req_to_token + req_pool_idx * pool_len
    
    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE