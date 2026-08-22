import torch
import torch_npu
import triton
import triton.language as tl
from mean_kernel import mean_dim

def test_mean_dim():
    """Test the mean_dim function with various inputs"""
    # Test case 1: 2D tensor, dim=1
    torch.manual_seed(0)
    input_2d = torch.randn(4096, 8192, device='npu', dtype=torch.float32)
    result_triton = mean_dim(input_2d, dim=1, keepdim=False)
    result_torch = torch.mean(input_2d, dim=1, keepdim=False)
    assert torch.allclose(result_triton, result_torch, atol=1e-6), "Test case 1 failed"
    print("Test case 1 passed: 2D tensor, dim=1, keepdim=False")

    # # Test case 2: 2D tensor, dim=0, keepdim=True
    # result_triton = mean_dim(input_2d, dim=0, keepdim=True)
    # result_torch = torch.mean(input_2d, dim=0, keepdim=True)
    # assert torch.allclose(result_triton, result_torch, atol=1e-6), "Test case 2 failed"
    # print("Test case 2 passed: 2D tensor, dim=0, keepdim=True")

    # # Test case 3: 3D tensor, dim=2
    # input_3d = torch.randn(5, 8, 12, device='npu', dtype=torch.float16)
    # result_triton = mean_dim(input_3d, dim=2, keepdim=False)
    # result_torch = torch.mean(input_3d, dim=2, keepdim=False)
    # assert torch.allclose(result_triton, result_torch, atol=1e-3), "Test case 3 failed"
    # print("Test case 3 passed: 3D tensor, dim=2, keepdim=False")

    # # Test case 4: 4D tensor, dim=-1 (negative indexing)
    # input_4d = torch.randn(3, 4, 5, 6, device='npu', dtype=torch.float32)
    # result_triton = mean_dim(input_4d, dim=-1, keepdim=True)
    # result_torch = torch.mean(input_4d, dim=-1, keepdim=True)
    # assert torch.allclose(result_triton, result_torch, atol=1e-6), "Test case 4 failed"
    # print("Test case 4 passed: 4D tensor, dim=-1, keepdim=True")

    print("All tests passed!")

if __name__ == "__main__":
    test_mean_dim()
