import torch
import triton
import triton.language as tl
import torch_npu

device = 'npu'

from copy_all_layer_kv_cache_tiled import *

def test_copy_all_layer_kv_cache_tiled():
    # Test parameters
    num_layers = 2
    cache_size = 1024
    num_locs = 3
    num_locs_upper = 4
    BYTES_PER_TILE = 64
    
    # Create test data on NPU
    data_buffers = []
    data_ptrs_list = []
    strides_list = []
    
    for i in range(num_layers):
        # Create a buffer for each layer
        buffer = torch.randint(0, 255, (cache_size, 128), dtype=torch.uint8, device=device)
        data_buffers.append(buffer)
        data_ptrs_list.append(buffer.data_ptr())
        strides_list.append(buffer.stride(0))
    
    # Create tensors for pointers and strides
    data_ptrs_tensor = torch.tensor(data_ptrs_list, dtype=torch.int64, device=device)
    strides_tensor = torch.tensor(strides_list, dtype=torch.int64, device=device)
    
    # Source and target locations
    src_loc = torch.tensor([10, 20, 30], dtype=torch.int64, device=device)
    tgt_loc = torch.tensor([100, 200, 300], dtype=torch.int64, device=device)
    
    # Save original values for comparison
    original_tgt_values = []
    for i in range(num_layers):
        buffer = data_buffers[i]
        tgt_values = []
        for loc in tgt_loc:
            tgt_values.append(buffer[loc.item(), :BYTES_PER_TILE].clone())
        original_tgt_values.append(tgt_values)
    
    # Launch kernel
    grid = (num_layers, triton.cdiv(strides_tensor.max().item(), BYTES_PER_TILE))
    copy_all_layer_kv_cache_tiled[grid](
        data_ptrs_tensor,
        strides_tensor,
        tgt_loc,
        src_loc,
        num_locs,
        num_locs_upper,
        BYTES_PER_TILE
    )
    
    # Verify results
    for i in range(num_layers):
        buffer = data_buffers[i]
        for j, (src, tgt) in enumerate(zip(src_loc, tgt_loc)):
            src_data = buffer[src.item(), :BYTES_PER_TILE]
            tgt_data = buffer[tgt.item(), :BYTES_PER_TILE]
            assert torch.equal(src_data, tgt_data), f"Layer {i}, location pair {j}: data mismatch"
    
    print("✅ copy_all_layer_kv_cache_tiled test passed")

if __name__ == "__main__":
    test_copy_all_layer_kv_cache_tiled()
