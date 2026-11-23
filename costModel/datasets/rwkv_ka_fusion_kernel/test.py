import torch
import triton
import triton.language as tl

from rwkv_ka_fusion_kernel import rwkv_ka_fusion


def torch_rwkv_ka_fusion(k, kk, a, ka, H, N):
    """Reference implementation in PyTorch"""
    if k.dim() == 1:
        T = 1
        C = k.shape[0]
    else:
        T, C = k.shape
    
    # Reshape tensors for easier processing if needed
    k_flat = k.view(-1, C) if k.dim() > 1 else k.unsqueeze(0)
    a_flat = a.view(-1, C) if a.dim() > 1 else a.unsqueeze(0)
    
    o_k_list = []
    o_kk_list = []
    o_kka_list = []
    
    for t in range(k_flat.shape[0]):
        k_t = k_flat[t]
        a_t = a_flat[t]
        
        o_k_frame = []
        o_kk_frame = []
        o_kka_frame = []
        
        for h in range(H):
            start_idx = h * N
            end_idx = (h + 1) * N
            
            k_h = k_t[start_idx:end_idx]
            a_h = a_t[start_idx:end_idx]
            kk_h = kk[start_idx:end_idx]
            ka_h = ka[start_idx:end_idx]
            
            # Compute okk
            kt = k_h * kk_h
            norm_kt = torch.sqrt(torch.sum(kt * kt) + 1e-12)
            okk_h = kt / norm_kt
            
            # Compute ok
            ok_h = k_h * (1 + (a_h - 1) * ka_h)
            
            # Compute okka
            okka_h = okk_h * a_h
            
            o_k_frame.append(ok_h)
            o_kk_frame.append(okk_h)
            o_kka_frame.append(okka_h)
        
        o_k_list.append(torch.cat(o_k_frame))
        o_kk_list.append(torch.cat(o_kk_frame))
        o_kka_list.append(torch.cat(o_kka_frame))
    
    o_k = torch.stack(o_k_list).view(k.shape)
    o_kk = torch.stack(o_kk_list).view(k.shape)
    o_kka = torch.stack(o_kka_list).view(k.shape)
    
    return o_k, o_kk, o_kka


def test_rwkv_ka_fusion():
    """Test the RWKV KA fusion kernel"""
    # Set device to NPU
    device = 'npu'
    
    # Test parameters
    T = 4
    H = 2
    N = 8
    C = H * N
    
    # Create test tensors
    torch.manual_seed(0)
    k = torch.randn((T, C), device=device, dtype=torch.float32)
    kk = torch.randn(C, device=device, dtype=torch.float32)
    a = torch.randn((T, C), device=device, dtype=torch.float32)
    ka = torch.randn(C, device=device, dtype=torch.float32)
    
    # Run Triton implementation
    o_k_triton, o_kk_triton, o_kka_triton = rwkv_ka_fusion(k, kk, a, ka, H, N)
    
    # Run PyTorch reference implementation
    o_k_torch, o_kk_torch, o_kka_torch = torch_rwkv_ka_fusion(k, kk, a, ka, H, N)
    
    # Check correctness
    assert torch.allclose(o_k_triton, o_k_torch, atol=1e-5, rtol=1e-3), "o_k mismatch"
    assert torch.allclose(o_kk_triton, o_kk_torch, atol=1e-5, rtol=1e-3), "o_kk mismatch"
    assert torch.allclose(o_kka_triton, o_kka_torch, atol=1e-5, rtol=1e-3), "o_kka mismatch"
    
    print("✅ RWKV KA Fusion tests passed!")
    print(f"o_k max difference: {torch.max(torch.abs(o_k_triton - o_k_torch))}")
    print(f"o_kk max difference: {torch.max(torch.abs(o_kk_triton - o_kk_torch))}")
    print(f"o_kka max difference: {torch.max(torch.abs(o_kka_triton - o_kka_torch))}")

if __name__ == "__main__":
    test_rwkv_ka_fusion()
    print("All tests passed!")



