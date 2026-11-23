import torch
import triton
import triton.language as tl
from post_reorder_triton_kernel_for_cutlass_moe import post_reorder_triton_for_cutlass_moe

device = 'npu'

def torch_post_reorder_reference(
    down_output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
):
    seq_len, topk = src2dst.shape
    hidden_size = down_output.shape[-1]
    output = torch.zeros((seq_len, hidden_size), dtype=down_output.dtype, device=down_output.device)
    
    for i in range(seq_len):
        for j in range(topk):
            expert_id = topk_ids[i, j].item()
            if expert_id != num_local_experts:
                dst_idx = src2dst[i, j].item()
                weight = topk_weights[i, j]
                output[i] += down_output[dst_idx] * weight
    return output

def test_post_reorder_triton_for_cutlass_moe():

    seq_len = 16
    topk = 4
    num_local_experts = 8
    hidden_size = 128

    torch.manual_seed(0)
    down_output = torch.randn((32, hidden_size), dtype=torch.float16, device=device)
    src2dst = torch.randint(0, 32, (seq_len, topk), dtype=torch.int32, device=device)
    topk_ids = torch.randint(0, num_local_experts + 1, (seq_len, topk), dtype=torch.int32, device=device)
    topk_weights = torch.rand((seq_len, topk), dtype=torch.float16, device=device)

    triton_output = post_reorder_triton_for_cutlass_moe(
        down_output, src2dst, topk_ids, topk_weights, num_local_experts
    )

    torch_output = torch_post_reorder_reference(
        down_output, src2dst, topk_ids, topk_weights, num_local_experts
    )

    print(f"Triton output: {triton_output}")
    print(f"PyTorch output: {torch_output}")
    print(f"Max difference: {torch.max(torch.abs(triton_output - torch_output))}")

    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2), "Outputs do not match!"
    print("✅ Triton and PyTorch implementations match!")

if __name__ == "__main__":
    test_post_reorder_triton_for_cutlass_moe()
