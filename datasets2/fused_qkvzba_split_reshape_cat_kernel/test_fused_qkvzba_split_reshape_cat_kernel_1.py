import torch
import pytest
import numpy as np
from fused_qkvzba_split_reshape_cat_kernel import fused_qkvzba_split_reshape_cat

def test_fused_qkvzba_split_reshape_cat():
    """Test the fused_qkvzba_split_reshape_cat function with NPU device."""
    device = 'npu'
    
    # Test parameters
    batch_size = 4
    num_heads_qk = 8
    num_heads_v = 16
    head_qk = 64
    head_v = 64
    
    # Dimensions for input tensors
    qkvz_dim_t = head_qk * 2 + num_heads_v // num_heads_qk * head_v * 2
    ba_dim_t = num_heads_v // num_heads_qk * 2
    
    # Create sample input tensors on NPU
    mixed_qkvz = torch.randn(batch_size, num_heads_qk * qkvz_dim_t, dtype=torch.float32, device=device)
    mixed_ba = torch.randn(batch_size, num_heads_qk * ba_dim_t, dtype=torch.float32, device=device)
    
    # Verify tensors are on NPU
    assert mixed_qkvz.device.type == 'npu'
    assert mixed_ba.device.type == 'npu'
    
    # Call the transformed function
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )
    
    # Verify output tensors are on NPU
    assert mixed_qkv.device.type == 'npu'
    assert z.device.type == 'npu'
    assert b.device.type == 'npu'
    assert a.device.type == 'npu'
    
    # Verify output shapes
    expected_qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    assert mixed_qkv.shape == (batch_size, expected_qkv_dim)
    assert z.shape == (batch_size, num_heads_v, head_v)
    assert b.shape == (batch_size, num_heads_v)
    assert a.shape == (batch_size, num_heads_v)
    
    # Create reference implementation using PyTorch operations
    def reference_implementation(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        batch_size = mixed_qkvz.size(0)
        heads_per_group = num_heads_v // num_heads_qk
        
        # Reshape to extract individual components
        mixed_qkvz_reshaped = mixed_qkvz.view(batch_size, num_heads_qk, -1)
        mixed_ba_reshaped = mixed_ba.view(batch_size, num_heads_qk, -1)
        
        # Extract Q, K, V, Z from mixed_qkvz
        q = mixed_qkvz_reshaped[:, :, :head_qk].contiguous().view(batch_size, num_heads_qk * head_qk)
        k = mixed_qkvz_reshaped[:, :, head_qk:head_qk*2].contiguous().view(batch_size, num_heads_qk * head_qk)
        v = mixed_qkvz_reshaped[:, :, head_qk*2:head_qk*2 + heads_per_group * head_v].contiguous().view(batch_size, num_heads_v * head_v)
        z = mixed_qkvz_reshaped[:, :, head_qk*2 + heads_per_group * head_v:].contiguous().view(batch_size, num_heads_v, head_v)
        
        # Extract B, A from mixed_ba
        b = mixed_ba_reshaped[:, :, :heads_per_group].contiguous().view(batch_size, num_heads_v)
        a = mixed_ba_reshaped[:, :, heads_per_group:].contiguous().view(batch_size, num_heads_v)
        
        # Concatenate Q, K, V for mixed_qkv output
        mixed_qkv = torch.cat([q, k, v], dim=1)
        
        return mixed_qkv, z, b, a
    
    # Get reference output
    ref_mixed_qkv, ref_z, ref_b, ref_a = reference_implementation(
        mixed_qkvz.cpu(), mixed_ba.cpu(), num_heads_qk, num_heads_v, head_qk, head_v
    )
    
    # Compare results with tolerance
    atol = 1e-5
    rtol = 1e-5
    
    torch.testing.assert_close(mixed_qkv.cpu(), ref_mixed_qkv, atol=atol, rtol=rtol)
    torch.testing.assert_close(z.cpu(), ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(b.cpu(), ref_b, atol=atol, rtol=rtol)
    torch.testing.assert_close(a.cpu(), ref_a, atol=atol, rtol=rtol)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_fused_qkvzba_split_reshape_cat()