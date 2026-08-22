import torch
import triton
import triton.language as tl
import torch_npu

def is_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()

from assign_draft_cache_locs_page_size_1 import *

def torch_assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len,
    topk,
    speculative_num_steps,
):
    copy_len = topk * speculative_num_steps
    for pid in range(req_pool_indices.shape[0]):
        out_cache_ptr = out_cache_loc[pid * topk * speculative_num_steps:(pid + 1) * topk * speculative_num_steps]
        kv_start = seq_lens[pid]
        token_pool_idx = req_pool_indices[pid]
        token_pool = req_to_token[token_pool_idx * pool_len:(token_pool_idx + 1) * pool_len]
        num_loop = (copy_len + 127) // 128
        for i in range(num_loop):
            copy_offset_start = i * 128
            copy_offset_end = min(copy_offset_start + 128, copy_len)
            data = token_pool[kv_start + copy_offset_start:kv_start + copy_offset_end]
            out_cache_ptr[copy_offset_start:copy_offset_end] = data

# Test function
def test_assign_draft_cache_locs():
    if not is_npu():
        print("NPU is not available, skipping test")
        return
        
    # Test parameters
    batch_size = 4
    pool_len = 1024
    topk = 3
    speculative_num_steps = 5
    
    # Create test data
    req_pool_indices = torch.randint(0, 10, (batch_size,), device='npu', dtype=torch.int32)
    req_to_token = torch.randint(0, 1000, (10 * pool_len,), device='npu', dtype=torch.int32)
    seq_lens = torch.randint(0, 100, (batch_size,), device='npu', dtype=torch.int32)
    out_cache_loc = torch.empty((batch_size * topk * speculative_num_steps,), device='npu', dtype=torch.int32)
    
    # Run Triton kernel
    grid = (batch_size,)
    assign_draft_cache_locs_page_size_1[grid](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        pool_len=pool_len,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
    )
    
    # Run reference implementation
    out_cache_loc_ref = torch.empty((batch_size * topk * speculative_num_steps,), device='npu', dtype=torch.int32)
    torch_assign_draft_cache_locs_page_size_1(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc_ref,
        pool_len,
        topk,
        speculative_num_steps,
    )
    
    # Compare results
    assert torch.allclose(out_cache_loc, out_cache_loc_ref), "Triton and Torch outputs do not match"
    print("✅ Triton and Torch match")
    print(f"Output cache locations: {out_cache_loc[:10]}...")

if __name__ == "__main__":
    test_assign_draft_cache_locs()
