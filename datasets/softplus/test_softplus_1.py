import logging
import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


from softplus_1 import softplus


# Test code
if __name__ == "__main__":
    # Create test input
    torch.manual_seed(0)
    x = torch.randn(512, device='npu', dtype=torch.float32)
    
    # Test parameters
    beta = 1.0
    threshold = 20.0
    
    # Compute with our implementation
    output_triton = softplus(x, beta, threshold)
    
    # Compute reference with PyTorch
    output_torch = torch.nn.functional.softplus(x, beta=beta, threshold=threshold)
    
    # Validate correctness
    print(f"Triton output: {output_triton}")
    print(f"PyTorch output: {output_torch}")
    print(f"Max difference: {torch.max(torch.abs(output_triton - output_torch))}")
    
    # Assertion to verify correctness
    assert torch.allclose(output_triton, output_torch, atol=1e-5, rtol=1e-3), "Outputs do not match!"
    print("✅ Softplus test passed!")
