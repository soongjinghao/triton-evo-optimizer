import torch
import triton
import triton.language as tl
import torch_npu

from fill_accepted_out_cache_loc import fill_accepted_out_cache_loc

device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream

def test_fill_accepted_out_cache_loc():
    # Test parameters
    size_upper = 128
    
    # Create test data on NPU
    accept_index = torch.randint(-1, 100, (size_upper,), dtype=torch.int64, device='npu')
    out_cache_loc = torch.randint(0, 1000, (100,), dtype=torch.int64, device='npu')
    accepted_out_cache_loc = torch.zeros(size_upper, dtype=torch.int64, device='npu')
    
    # Launch kernel
    grid = (size_upper,)
    fill_accepted_out_cache_loc[grid](
        accept_index, 
        out_cache_loc, 
        accepted_out_cache_loc, 
        size_upper
    )
    
    # Reference implementation
    accept_index_cpu = accept_index.cpu()
    out_cache_loc_cpu = out_cache_loc.cpu()
    accepted_out_cache_loc_ref = torch.zeros(size_upper, dtype=torch.int64)
    
    for pid in range(size_upper):
        masks = (accept_index_cpu[:pid] != -1).to(torch.int64)
        dst = torch.sum(masks).item()
        src = accept_index_cpu[pid].item()
        if src > -1:
            value = out_cache_loc_cpu[src].item()
            accepted_out_cache_loc_ref[dst] = value
    
    # Compare results
    assert torch.allclose(accepted_out_cache_loc.cpu(), accepted_out_cache_loc_ref), \
        "Kernel output does not match reference implementation"
    
    print("✅ fill_accepted_out_cache_loc test passed")

if __name__ == "__main__":
    test_fill_accepted_out_cache_loc()
