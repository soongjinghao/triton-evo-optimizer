import torch
import triton
import triton.language as tl


# Device configuration
DEVICE = 'npu'  # Changed from 'cuda' to 'npu'

from fn_triton_kernel import *

# ========== Test Cases ==========

def test_fn_triton_basic():
    """Test basic functionality with small tensors"""
    # Setup
    num_tokens = 128
    num_local_heads = 8
    qk_nope_head_dim = 64
    qk_rope_head_dim = 32
    
    # Create input tensors on NPU
    k_nope = torch.randn(num_tokens, num_local_heads, qk_nope_head_dim, 
                         device='npu', dtype=torch.float32)
    k_rope = torch.randn(num_tokens, qk_rope_head_dim, 
                         device='npu', dtype=torch.float32)
    k = torch.zeros(num_tokens, num_local_heads, qk_nope_head_dim + qk_rope_head_dim,
                    device='npu', dtype=torch.float32)
    
    # Run kernel
    fn_triton(k, k_nope, k_rope, qk_nope_head_dim, qk_rope_head_dim, num_local_heads)
    
    # Verify results - construct expected output manually
    expected = torch.zeros_like(k)
    # Copy nope values
    expected[:, :, :qk_nope_head_dim] = k_nope
    # Copy rope values (broadcasted across heads)
    expected[:, :, qk_nope_head_dim:] = k_rope.unsqueeze(1).expand(-1, num_local_heads, -1)
    
    assert torch.allclose(k, expected, rtol=1e-5, atol=1e-5), \
        f"Output mismatch: max diff = {(k - expected).abs().max().item()}"
    print("? Basic test passed")

if __name__ == "__main__":
    test_fn_triton_basic()




