import torch
import triton
import triton.language as tl
import torch_npu

device = 'npu'

from assign_extend_cache_locs import *

def run_assign_extend_cache_locs(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    grid = (req_pool_indices_tensor.numel(),)
    assign_extend_cache_locs[grid](
        req_pool_indices_tensor,
        req_to_token_tensor,
        start_offset_tensor,
        end_offset_tensor,
        out_cache_loc_tensor,
        pool_len=pool_len,
        bs_upper=bs_upper
    )
    return out_cache_loc_tensor

def torch_reference_impl(
    req_pool_indices_tensor,
    req_to_token_tensor,
    start_offset_tensor,
    end_offset_tensor,
    out_cache_loc_tensor,
    pool_len,
    bs_upper
):
    out_cache_loc_ref = out_cache_loc_tensor.clone()
    for pid in range(req_pool_indices_tensor.numel()):
        kv_start = start_offset_tensor[pid].item()
        kv_end = end_offset_tensor[pid].item()
        req_idx = req_pool_indices_tensor[pid].item()
        
        # Calculate out_offset
        out_offset = 0
        for i in range(pid):
            out_offset += end_offset_tensor[i].item() - start_offset_tensor[i].item()
        
        # Copy data
        for i in range(kv_start, kv_end):
            token_idx = req_to_token_tensor[req_idx * pool_len + i].item()
            out_cache_loc_ref[out_offset + (i - kv_start)] = token_idx
            
    return out_cache_loc_ref

if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    pool_len = 128
    bs_upper = 8
    
    # Create test data
    req_pool_indices = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.int32)
    req_to_token = torch.randint(0, 1000, (10 * pool_len,), device=device, dtype=torch.int32)
    start_offset = torch.randint(0, 64, (batch_size,), device=device, dtype=torch.int32)
    end_offset = start_offset + torch.randint(1, 32, (batch_size,), device=device, dtype=torch.int32)
    out_cache_loc = torch.zeros(256, device=device, dtype=torch.int32)
    
    # Run Triton kernel
    out_cache_loc_triton = run_assign_extend_cache_locs(
        req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc.clone(), pool_len, bs_upper
    )
    
    # Run reference implementation
    out_cache_loc_ref = torch_reference_impl(
        req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc.clone(), pool_len, bs_upper
    )
    
    # Validate results
    assert torch.allclose(out_cache_loc_triton, out_cache_loc_ref), "Results do not match!"
    print("✅ Triton kernel output matches reference implementation")
    print(f"Triton result: {out_cache_loc_triton[:20]}")
    print(f"Reference result: {out_cache_loc_ref[:20]}")