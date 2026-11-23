import torch

from recompute_w_u_fwd_kernel import recompute_w_u_fwd

device = torch.device("npu")

def test_recompute_w_u_fwd():

    B, T, H, K, V = 2, 32, 4, 64, 32
    BT = 16
    
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32)
    g_cumsum = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)

    def ref_recompute_w_u_fwd(k, v, beta, g_cumsum, A):
        B, T, H, K = k.shape
        V = v.shape[-1]
        BT = A.shape[-1]
        
        u = torch.zeros_like(v)
        w = torch.zeros(B, T, H, K, device=v.device, dtype=v.dtype)
        
        for b in range(B):
            for h in range(H):
                for t in range(0, T, BT):
                    end_t = min(t + BT, T)
                    bt_size = end_t - t

                    beta_block = beta[b, t:end_t, h].unsqueeze(1)
                    v_block = v[b, t:end_t, h, :]
                    v_beta = v_block * beta_block
                    A_block = A[b, t:end_t, h, :bt_size]
                    u_block = torch.matmul(A_block, v_beta)
                    u[b, t:end_t, h, :] = u_block

                    k_block = k[b, t:end_t, h, :]
                    g_block = g_cumsum[b, t:end_t, h].unsqueeze(1)
                    exp_g = torch.exp(g_block)
                    k_beta_g = k_block * beta_block * exp_g
                    w_block = torch.matmul(A_block, k_beta_g)
                    w[b, t:end_t, h, :] = w_block
        
        return w, u
    
    ref_w, ref_u = ref_recompute_w_u_fwd(k, v, beta, g_cumsum, A)
    test_w, test_u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, None)
    
    assert torch.allclose(ref_w, test_w, rtol=1e-4, atol=1e-4), "w output mismatch"
    assert torch.allclose(ref_u, test_u, rtol=1e-4, atol=1e-4), "u output mismatch"

    B, T, H, K, V = 1, 64, 8, 128, 64
    BT = 32
    
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32)
    g_cumsum = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)
    
    ref_w, ref_u = ref_recompute_w_u_fwd(k, v, beta, g_cumsum, A)
    test_w, test_u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, None)
    
    assert torch.allclose(ref_w, test_w, rtol=1e-4, atol=1e-4), "w output mismatch (test 2)"
    assert torch.allclose(ref_u, test_u, rtol=1e-4, atol=1e-4), "u output mismatch (test 2)"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_recompute_w_u_fwd()
