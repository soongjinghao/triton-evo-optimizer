import torch
from l2norm_fwd_kernel1 import l2norm_fwd

def test_l2norm():

    torch.manual_seed(0)
    x = torch.randn(64, 64, device='npu', dtype=torch.float32)
    eps = 1e-6

    y_triton = l2norm_fwd(x, eps)

    x_reshaped = x.view(-1, x.shape[-1])
    norm = torch.sqrt(torch.sum(x_reshaped ** 2, dim=-1, keepdim=True))
    y_ref = x_reshaped / (norm + eps)
    y_ref = y_ref.view(x.shape)

    assert torch.allclose(y_triton, y_ref, atol=1e-5, rtol=1e-3), f"Max diff: {torch.max(torch.abs(y_triton - y_ref))}"
    print("L2Norm test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_triton.shape}")
    print(f"Max difference: {torch.max(torch.abs(y_triton - y_ref))}")

if __name__ == "__main__":
    test_l2norm()
