import torch
import triton
import triton.language as tl
import torch_npu

from l2norm_fwd_kernel2 import l2norm_fwd

def test_l2norm_fwd():
    # Test case 1: Basic functionality
    torch.manual_seed(0)
    x = torch.randn(128, 64, device='npu', dtype=torch.float32)
    eps = 1e-6
    
    # Compute with our Triton kernel
    y_triton = l2norm_fwd(x, eps)
    
    # Compute reference with PyTorch
    x_reshaped = x.view(-1, x.shape[-1])
    square_sum = torch.sum(x_reshaped * x_reshaped, dim=1, keepdim=True)
    rsqrt = torch.rsqrt(square_sum + eps)
    y_torch = x_reshaped * rsqrt
    y_torch = y_torch.view(x.shape)
    
    # Check correctness
    assert torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-3), f"Max diff: {torch.max(torch.abs(y_triton - y_torch))}"
    print("✅ Test 1 passed: Basic L2 normalization")
    
    # print("All tests passed!")

if __name__ == "__main__":
    test_l2norm_fwd()
