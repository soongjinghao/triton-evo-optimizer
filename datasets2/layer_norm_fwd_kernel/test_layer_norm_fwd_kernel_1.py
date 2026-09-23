import torch
import triton
import pytest
import numpy as np

from layer_norm_fwd_kernel import *

def test_layer_norm_fwd_basic():
    """Test basic layer normalization forward pass."""
    torch.manual_seed(42)
    
    # Test parameters
    M, N = 8192, 2048  # Batch size, feature dimension
    eps = 1e-5
    device = torch.device('npu')
    
    # Create random input
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    
    # Create weight and bias
    weight = torch.randn(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)
    
    # Run custom layer norm
    out_custom, mean_custom, rstd_custom = layer_norm_fwd(
        x=x,
        weight=weight,
        bias=bias,
        eps=eps,
        z=None,
        out=None,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False
    )
    
    # Compute reference using PyTorch's layer_norm
    # Note: PyTorch's layer_norm expects normalized shape as the last dimensions
    out_ref = torch.nn.functional.layer_norm(
        x, 
        normalized_shape=[N], 
        weight=weight, 
        bias=bias, 
        eps=eps
    )
    
    # Verify output matches reference
    assert torch.allclose(out_custom, out_ref, rtol=1e-4, atol=1e-5), \
        f"Output mismatch: max diff = {torch.max(torch.abs(out_custom - out_ref))}"
    
    # Verify mean and std computations
    # For each row, compute reference mean and std
    mean_ref = x.mean(dim=1)
    var_ref = x.var(dim=1, unbiased=False)
    rstd_ref = 1.0 / torch.sqrt(var_ref + eps)
    
    # Since our implementation might compute mean/rstd per group, 
    # we need to check the first group (group_size = N in this case)
    assert torch.allclose(mean_custom[:M], mean_ref, rtol=1e-4, atol=1e-5), \
        "Mean computation mismatch"
    assert torch.allclose(rstd_custom[:M], rstd_ref, rtol=1e-4, atol=1e-5), \
        "Rstd computation mismatch"
    
    print("✓ Basic layer norm test passed")
    
    # # Test RMS normalization
    # print("\nTesting RMS normalization...")
    # out_rms, _, rstd_rms = layer_norm_fwd(
    #     x=x,
    #     weight=weight,
    #     bias=bias,
    #     eps=eps,
    #     z=None,
    #     out=None,
    #     group_size=None,
    #     norm_before_gate=True,
    #     is_rms_norm=True
    # )
    
    # # Reference RMS norm: normalize by root mean square
    # rms_ref = torch.sqrt(torch.mean(x**2, dim=1) + eps)
    # x_hat_rms_ref = x / rms_ref.unsqueeze(1)
    # out_rms_ref = x_hat_rms_ref * weight + bias
    
    # assert torch.allclose(out_rms, out_rms_ref, rtol=1e-4, atol=1e-5), \
    #     f"RMS norm mismatch: max diff = {torch.max(torch.abs(out_rms - out_rms_ref))}"
    
    # print("✓ RMS normalization test passed")
    
    # # Test with group normalization
    # print("\nTesting group normalization...")
    # group_size = 128
    # out_group, mean_group, rstd_group = layer_norm_fwd(
    #     x=x,
    #     weight=weight,
    #     bias=bias,
    #     eps=eps,
    #     z=None,
    #     out=None,
    #     group_size=group_size,
    #     norm_before_gate=True,
    #     is_rms_norm=False
    # )
    
    # # For group norm, verify each group separately
    # ngroups = N // group_size
    # for g in range(ngroups):
    #     start_idx = g * group_size
    #     end_idx = (g + 1) * group_size
        
    #     # Reference computation for this group
    #     x_group = x[:, start_idx:end_idx]
    #     weight_group = weight[start_idx:end_idx]
    #     bias_group = bias[start_idx:end_idx]
        
    #     out_group_ref = torch.nn.functional.layer_norm(
    #         x_group,
    #         normalized_shape=[group_size],
    #         weight=weight_group,
    #         bias=bias_group,
    #         eps=eps
    #     )
        
    #     # Compare corresponding output group
    #     out_custom_group = out_group[:, start_idx:end_idx]
        
    #     assert torch.allclose(out_custom_group, out_group_ref, rtol=1e-4, atol=1e-5), \
    #         f"Group {g} mismatch: max diff = {torch.max(torch.abs(out_custom_group - out_group_ref))}"
    
    # print("✓ Group normalization test passed")
    
    # # Test with gating (z tensor)
    # print("\nTesting with gating...")
    # z = torch.randn_like(x)
    
    # # Test norm_before_gate=True
    # out_gate_before, _, _ = layer_norm_fwd(
    #     x=x,
    #     weight=weight,
    #     bias=bias,
    #     eps=eps,
    #     z=z,
    #     out=None,
    #     group_size=None,
    #     norm_before_gate=True,
    #     is_rms_norm=False
    # )
    
    # # Test norm_before_gate=False
    # out_gate_after, _, _ = layer_norm_fwd(
    #     x=x,
    #     weight=weight,
    #     bias=bias,
    #     eps=eps,
    #     z=z,
    #     out=None,
    #     group_size=None,
    #     norm_before_gate=False,
    #     is_rms_norm=False
    # )
    
    # # Verify they're different (gating applied at different stages)
    # assert not torch.allclose(out_gate_before, out_gate_after, rtol=1e-4, atol=1e-5), \
    #     "norm_before_gate should produce different results"
    
    # print("✓ Gating test passed")
    
    # # Test edge cases
    # print("\nTesting edge cases...")
    
    # # Small batch size
    # x_small = torch.randn(1, N, device=device, dtype=torch.float32)
    # out_small, _, _ = layer_norm_fwd(
    #     x=x_small,
    #     weight=weight,
    #     bias=bias,
    #     eps=eps,
    #     z=None,
    #     out=None,
    #     group_size=None,
    #     norm_before_gate=True,
    #     is_rms_norm=False
    # )
    
    # out_small_ref = torch.nn.functional.layer_norm(
    #     x_small, 
    #     normalized_shape=[N], 
    #     weight=weight, 
    #     bias=bias, 
    #     eps=eps
    # )
    
    # assert torch.allclose(out_small, out_small_ref, rtol=1e-4, atol=1e-5), \
    #     "Small batch test failed"
    
    # print("✓ Small batch test passed")
    
    # # Test without bias
    # print("\nTesting without bias...")
    # out_no_bias, _, _ = layer_norm_fwd(
    #     x=x,
    #     weight=weight,
    #     bias=None,
    #     eps=eps,
    #     z=None,
    #     out=None,
    #     group_size=None,
    #     norm_before_gate=True,
    #     is_rms_norm=False
    # )
    
    # out_no_bias_ref = torch.nn.functional.layer_norm(
    #     x, 
    #     normalized_shape=[N], 
    #     weight=weight, 
    #     bias=None, 
    #     eps=eps
    # )
    
    # assert torch.allclose(out_no_bias, out_no_bias_ref, rtol=1e-4, atol=1e-5), \
    #     "No bias test failed"
    
    # print("✓ No bias test passed")
    
    # print("\n✅ All layer_norm_fwd tests passed!")

# Helper function (needs to be defined or imported)
def next_power_of_2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1

def cdiv(a, b):
    return (a + b - 1) // b

if __name__ == "__main__":
    # Run the test
    test_layer_norm_fwd_basic()

