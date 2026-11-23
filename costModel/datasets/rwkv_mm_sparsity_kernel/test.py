import torch
import triton
import triton.language as tl
from rwkv_mm_sparsity_kernel import rwkv_mm_sparsity

def test_rwkv_mm_sparsity():

    torch.manual_seed(0)
    k_size = 1024
    v_cols = 512

    k = torch.randn(k_size, device='npu', dtype=torch.float32)
    v = torch.randn(k_size, v_cols, device='npu', dtype=torch.float32)

    k_sparse = k.clone()
    k_sparse[::2] = 0

    triton_output = rwkv_mm_sparsity(k_sparse, v)

    torch_output = torch.matmul(k_sparse, v)

    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3), \
        f"Max difference: {torch.max(torch.abs(triton_output - torch_output))}"
    
    print("✅ RWKV MM Sparsity test passed!")
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Max difference: {torch.max(torch.abs(triton_output - torch_output))}")

if __name__ == "__main__":
    test_rwkv_mm_sparsity()
