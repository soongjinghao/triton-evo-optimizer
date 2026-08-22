import torch
import triton
import pytest

from compute_identity_kernel import zero_experts_compute_triton

def test_zero_experts_compute_triton_identity():
    """Test the zero_experts_compute_triton function with identity zero expert type."""
    # Set device to NPU
    device = torch.device('npu')
    
    # Create sample input tensors on NPU
    num_tokens = 4096
    hidden_dim = 512
    num_experts = 8
    top_k = 2
    
    # Generate random inputs
    torch.manual_seed(42)  # For reproducibility
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device)
    expert_indices = torch.randint(0, num_experts * 2, (num_tokens, top_k), device=device)
    expert_scales = torch.randn(num_tokens, top_k, device=device)
    
    # Call the transformed kernel function
    output = zero_experts_compute_triton(
        expert_indices=expert_indices,
        expert_scales=expert_scales,
        num_experts=num_experts,
        zero_expert_type="identity",
        hidden_states=hidden_states
    )
    
    # Validate correctness against PyTorch reference implementation
    # Reference implementation: identity zero expert type logic
    zero_expert_mask = expert_indices < num_experts
    zero_expert_scales = expert_scales.clone()
    zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices_ref = expert_indices.clone()
    expert_scales_ref = expert_scales.clone()
    expert_indices_ref[normal_expert_mask] = 0
    expert_scales_ref[normal_expert_mask] = 0.0
    
    # Compute expected output
    expected_output = torch.zeros_like(hidden_states)
    for i in range(num_tokens):
        for j in range(top_k):
            scale = zero_expert_scales[i, j]
            expert_idx = expert_indices_ref[i, j]
            if expert_idx == 0:  # Only process zero experts
                expected_output[i] += hidden_states[i] * scale
    
    # Verify the output matches expected results
    assert output.shape == hidden_states.shape, f"Output shape {output.shape} != expected {hidden_states.shape}"
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected result"
    assert output.device.type == 'npu', f"Output not on NPU device: {output.device}"

if __name__ == "__main__":
    # Run the tests
    test_zero_experts_compute_triton_identity()
    print("All tests passed!")
