from typing import Tuple

import torch
import triton
import triton.language as tl

from experts_combine_kernel import *

def test_experts_combine():
    # Test case 1: Basic functionality with combine_k > 1
    torch.manual_seed(0)
    device = 'npu'
    
    bs, hidden_dim, combine_k = 4, 1024, 2
    moe_hidden_states = torch.randn(bs, combine_k, hidden_dim, device=device, dtype=torch.float32)
    mlp_hidden_states = torch.randn(bs, hidden_dim, device=device, dtype=torch.float32)
    
    # Compute using triton kernel
    triton_output = experts_combine_triton(moe_hidden_states, mlp_hidden_states)
    
    # Compute reference using PyTorch
    moe_sum = moe_hidden_states.sum(dim=1)  # Sum over combine_k dimension
    torch_output = (moe_sum + mlp_hidden_states) / 1.4142135623730951
    
    # Verify correctness
    assert torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-3), \
        f"Max difference: {torch.max(torch.abs(triton_output - torch_output))}"
    
    print("Test case 1 passed: Basic functionality with combine_k > 1")
    
    # Test case 2: Pre-combined case (combine_k = 1)
    moe_hidden_states_2d = torch.randn(bs, hidden_dim, device=device, dtype=torch.float32)
    triton_output_2d = experts_combine_triton(moe_hidden_states_2d, mlp_hidden_states)
    
    torch_output_2d = (moe_hidden_states_2d + mlp_hidden_states) / 1.4142135623730951
    
    assert torch.allclose(triton_output_2d, torch_output_2d, atol=1e-5, rtol=1e-3), \
        f"Max difference: {torch.max(torch.abs(triton_output_2d - torch_output_2d))}"
    
    print("Test case 2 passed: Pre-combined case (combine_k = 1)")
    
    # Test case 3: With output buffer
    output_buffer = torch.empty(bs * hidden_dim, device=device, dtype=torch.float32)
    triton_output_buffer = experts_combine_triton(moe_hidden_states, mlp_hidden_states, output_buffer)
    
    assert torch.allclose(triton_output_buffer, torch_output, atol=1e-5, rtol=1e-3), \
        f"Max difference: {torch.max(torch.abs(triton_output_buffer - torch_output))}"
    
    print("Test case 3 passed: With output buffer")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_experts_combine()
