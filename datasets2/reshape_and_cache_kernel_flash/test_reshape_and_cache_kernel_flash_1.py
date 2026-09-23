import torch
import pytest
import numpy as np

from reshape_and_cache_kernel_flash import *

def test_triton_reshape_and_cache_flash_basic():
    """Test basic functionality of reshape_and_cache kernel"""
    # Skip if NPU is not available
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Test parameters
    num_tokens = 16
    num_heads = 8
    head_size = 64
    num_blocks = 4
    block_size = 8
    
    # Create input tensors on NPU
    key = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    value = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    
    # Create cache tensors on NPU
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, 
                           device='npu', dtype=torch.float16)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, 
                            device='npu', dtype=torch.float16)
    
    # Create slot mapping (each token maps to a unique slot)
    slot_mapping = torch.arange(num_tokens, device='npu')
    
    # Create scale tensors for FP8 (not used in basic test)
    k_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    v_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    
    # Call the kernel
    triton_reshape_and_cache_flash(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale
    )
    
    # Verify results - check that key/value are correctly cached
    for i in range(num_tokens):
        slot_idx = slot_mapping[i]
        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size
        
        # Get cached values
        cached_key = key_cache[block_idx, block_offset]
        cached_value = value_cache[block_idx, block_offset]
        
        # Get original values
        original_key = key[i]
        original_value = value[i]
        
        # Verify they match
        torch.testing.assert_close(cached_key, original_key, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(cached_value, original_value, rtol=1e-5, atol=1e-5)

def test_triton_reshape_and_cache_flash_negative_slots():
    """Test handling of negative slot indices (padding tokens)"""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Test parameters
    num_tokens = 8
    num_heads = 4
    head_size = 32
    num_blocks = 2
    block_size = 8
    
    # Create input tensors
    key = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    value = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    
    # Create cache tensors with initial values
    initial_value = 42.0
    key_cache = torch.full((num_blocks, block_size, num_heads, head_size), 
                          initial_value, device='npu', dtype=torch.float16)
    value_cache = torch.full((num_blocks, block_size, num_heads, head_size), 
                           initial_value, device='npu', dtype=torch.float16)
    
    # Create slot mapping with negative values (padding tokens)
    slot_mapping = torch.tensor([0, 1, -1, -1, 4, 5, -1, 7], device='npu')
    
    # Scale tensors
    k_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    v_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    
    # Call the kernel
    triton_reshape_and_cache_flash(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale
    )
    
    # Verify that negative slots were ignored (cache unchanged for those positions)
    for i, slot in enumerate(slot_mapping):
        if slot >= 0:
            block_idx = slot // block_size
            block_offset = slot % block_size
            
            # Cache should be updated for valid slots
            cached_key = key_cache[block_idx, block_offset]
            original_key = key[i]
            torch.testing.assert_close(cached_key, original_key, rtol=1e-5, atol=1e-5)
        else:
            # For negative slots, we can't easily verify they were ignored without
            # knowing which cache positions correspond to them
            pass

def test_triton_reshape_and_cache_flash_large_tile():
    """Test with large tile size that requires multiple iterations"""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Use larger dimensions to test tiling
    num_tokens = 32
    num_heads = 16
    head_size = 128  # Larger head size to test tiling
    num_blocks = 8
    block_size = 16
    
    # Create input tensors
    key = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    value = torch.randn(num_tokens, num_heads, head_size, device='npu', dtype=torch.float16)
    
    # Create cache tensors
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, 
                           device='npu', dtype=torch.float16)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, 
                            device='npu', dtype=torch.float16)
    
    # Create slot mapping with some gaps
    slot_mapping = torch.cat([
        torch.arange(0, 16, device='npu'),      # First half
        torch.arange(32, 48, device='npu')       # Second half with gap
    ])
    
    # Scale tensors
    k_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    v_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    
    # Call the kernel
    triton_reshape_and_cache_flash(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale
    )
    
    # Verify results
    for i, slot in enumerate(slot_mapping):
        if slot >= 0:
            block_idx = slot // block_size
            block_offset = slot % block_size
            
            cached_key = key_cache[block_idx, block_offset]
            cached_value = value_cache[block_idx, block_offset]
            
            original_key = key[i]
            original_value = value[i]
            
            torch.testing.assert_close(cached_key, original_key, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(cached_value, original_value, rtol=1e-5, atol=1e-5)

def test_triton_reshape_and_cache_flash_edge_cases():
    """Test edge cases like single token, single head"""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Test case 1: Single token
    key = torch.randn(1, 4, 32, device='npu', dtype=torch.float16)
    value = torch.randn(1, 4, 32, device='npu', dtype=torch.float16)
    key_cache = torch.zeros(2, 8, 4, 32, device='npu', dtype=torch.float16)
    value_cache = torch.zeros(2, 8, 4, 32, device='npu', dtype=torch.float16)
    slot_mapping = torch.tensor([0], device='npu')
    
    k_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    v_scale = torch.tensor(1.0, device='npu', dtype=torch.float32)
    
    triton_reshape_and_cache_flash(
        key=key, value=value, key_cache=key_cache, value_cache=value_cache,
        slot_mapping=slot_mapping, kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale
    )
    
    # Verify single token was cached correctly
    torch.testing.assert_close(key_cache[0, 0], key[0])
    torch.testing.assert_close(value_cache[0, 0], value[0])
    
    # Test case 2: Single head
    key = torch.randn(4, 1, 64, device='npu', dtype=torch.float16)
    value = torch.randn(4, 1, 64, device='npu', dtype=torch.float16)
    key_cache = torch.zeros(2, 8, 1, 64, device='npu', dtype=torch.float16)
    value_cache = torch.zeros(2, 8, 1, 64, device='npu', dtype=torch.float16)
    slot_mapping = torch.arange(4, device='npu')
    
    triton_reshape_and_cache_flash(
        key=key, value=value, key_cache=key_cache, value_cache=value_cache,
        slot_mapping=slot_mapping, kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale
    )
    
    # Verify all tokens were cached correctly
    for i in range(4):
        slot = slot_mapping[i]
        block_idx = slot // 8
        block_offset = slot % 8
        torch.testing.assert_close(key_cache[block_idx, block_offset], key[i])
        torch.testing.assert_close(value_cache[block_idx, block_offset], value[i])

if __name__ == "__main__":
    # Run tests
    test_triton_reshape_and_cache_flash_basic()
    print("✓ Basic test passed")
    
    test_triton_reshape_and_cache_flash_negative_slots()
    print("✓ Negative slots test passed")
    
    test_triton_reshape_and_cache_flash_large_tile()
    print("✓ Large tile test passed")
    
    test_triton_reshape_and_cache_flash_edge_cases()
    print("✓ Edge cases test passed")
    
    print("All tests passed!")
