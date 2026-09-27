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
    BLOCK_REQUESTS: tl.constexpr = 4
    
    pid = tl.program_id(axis=0)
    base_idx = pid * BLOCK_REQUESTS
    
    # 批量加载多个请求的start和end
    offsets = tl.arange(0, BLOCK_REQUESTS)
    req_indices = base_idx + offsets
    mask_req = req_indices < bs_upper
    
    kv_starts = tl.load(start_offset + req_indices, mask=mask_req, other=0)
    kv_ends = tl.load(end_offset + req_indices, mask=mask_req, other=0)
    
    # 计算每个请求的out_offset(前缀和)
    # 对于第i个请求,out_offset = sum_{j<base_idx+i} (end_j - start_j)
    # 这里我们只计算当前块内的前缀和,全局前缀和需要从外部传入或计算
    # 由于无法直接获取全局前缀和,我们保持原有逻辑但向量化处理
    lengths = kv_ends - kv_starts
    
    # 计算每个请求的out_offset(相对于当前块起始)
    # 使用前缀和计算
    prefix_sum = tl.cumsum(lengths, axis=0) - lengths
    
    # 循环处理每个请求
    for i in range(BLOCK_REQUESTS):
        valid = mask_req & (i < BLOCK_REQUESTS)
        kv_start = tl.load(start_offset + base_idx + i, mask=valid, other=0)
        kv_end = tl.load(end_offset + base_idx + i, mask=valid, other=0)
        
        # 计算out_offset
        length_offset = tl.arange(0, bs_upper)
        start = tl.load(start_offset + length_offset, mask=length_offset < base_idx + i, other=0)
        end = tl.load(end_offset + length_offset, mask=length_offset < base_idx + i, other=0)
        out_offset = tl.sum(end - start, axis=0)
        
        token_pool = req_to_token + tl.load(req_pool_indices + base_idx + i, mask=valid, other=0) * pool_len
        out_cache_ptr = out_cache_loc + out_offset
        
        load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
        save_offset = tl.arange(0, BLOCK_SIZE)
        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        
        for _ in range(num_loop):
            mask = load_offset < kv_end
            data = tl.load(token_pool + load_offset, mask=mask)
            tl.store(out_cache_ptr + save_offset, data, mask=mask)
            load_offset += BLOCK_SIZE
            save_offset += BLOCK_SIZE