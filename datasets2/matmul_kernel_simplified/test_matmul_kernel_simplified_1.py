

import torch
import numpy as np

from matmul_kernel_simplified import matmul_persistent

def test_matmul_persistent_basic():
    """Test basic matrix multiplication functionality"""
    device = 'npu'
    
    # Create test tensors on NPU
    M, K, N = 64, 128, 32
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)
    
    # Test with bias
    result = matmul_persistent(a, b, bias)
    
    # Reference implementation
    expected = torch.matmul(a, b) + bias
    
    # Verify correctness
    assert result.device.type == 'npu', "Output tensor should be on NPU"
    assert result.shape == (M, N), f"Expected shape {(M, N)}, got {result.shape}"
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    
    print("✓ Basic matmul with bias test passed")

def test_matmul_persistent_no_bias():
    """Test matrix multiplication without bias"""
    device = 'npu'

    M, K, N = 128, 256, 64
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    result = matmul_persistent(a, b, None)

    expected = torch.matmul(a, b)

    assert result.device.type == 'npu', "Output tensor should be on NPU"
    assert result.shape == (M, N), f"Expected shape {(M, N)}, got {result.shape}"
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    
    print("✓ Matmul without bias test passed")

def test_matmul_persistent_float32():
    """Test matrix multiplication with float32 precision"""
    device = 'npu'
    
    # Create test tensors on NPU
    M, K, N = 32, 64, 48
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    
    # Test with float32
    result = matmul_persistent(a, b, None)
    
    # Reference implementation
    expected = torch.matmul(a, b)
    
    # Verify correctness
    assert result.device.type == 'npu', "Output tensor should be on NPU"
    assert result.shape == (M, N), f"Expected shape {(M, N)}, got {result.shape}"
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    print("✓ Float32 matmul test passed")

if __name__ == "__main__":
    # test_matmul_persistent_basic()
    # test_matmul_persistent_no_bias()
    test_matmul_persistent_float32()
    print("All tests passed!")