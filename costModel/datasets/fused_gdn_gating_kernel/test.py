import torch
from fused_gdn_gating_kernel import fused_gdn_gating

def test_fused_gdn_gating():

    device = torch.device('npu')
    batch, num_heads = 2048, 256
    
    A_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    a = torch.randn(batch, num_heads, device=device, dtype=torch.float32)
    b = torch.randn(batch, num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device=device, dtype=torch.float32)

    g_ref = -torch.exp(A_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias)
    beta_output_ref = torch.sigmoid(b.float())

    g, beta_output = fused_gdn_gating(A_log, a, b, dt_bias)

    assert torch.allclose(g.squeeze(0), g_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(beta_output.squeeze(0), beta_output_ref.to(torch.float32), rtol=1e-5, atol=1e-5)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_fused_gdn_gating()
