import torch
import triton
import triton.language as tl

@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    blk_size: tl.constexpr,
    k_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_program: tl.constexpr = 2
    base_col = pid * block_size * num_blocks_per_program
    col_offsets = tl.arange(0, block_size)
    acc0 = tl.zeros((block_size,), dtype=tl.float32)
    acc1 = tl.zeros((block_size,), dtype=tl.float32)
    for i in range(0, tl.cdiv(k_size, blk_size)):
        k_offset = i * blk_size + tl.arange(0, blk_size)
        k_mask = k_offset < k_size
        k = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
        k_nonzero_mask = k != 0
        col_idx0 = base_col + col_offsets
        col_mask0 = col_idx0 < v_cols
        v_block0 = tl.load(v_ptr + k_offset[:, None] * v_cols + col_idx0[None, :],
                           mask=k_mask[:, None] & col_mask0[None, :], other=0.0)
        k_broadcast0 = tl.broadcast_to(k[:, None], (blk_size, block_size))
        weighted_v0 = tl.where(k_nonzero_mask[:, None] & col_mask0[None, :],
                               k_broadcast0.to(tl.float32) * v_block0.to(tl.float32), 0.0)
        acc0 += tl.sum(weighted_v0, axis=0)
        col_idx1 = base_col + block_size + col_offsets
        col_mask1 = col_idx1 < v_cols
        v_block1 = tl.load(v_ptr + k_offset[:, None] * v_cols + col_idx1[None, :],
                           mask=k_mask[:, None] & col_mask1[None, :], other=0.0)
        k_broadcast1 = tl.broadcast_to(k[:, None], (blk_size, block_size))
        weighted_v1 = tl.where(k_nonzero_mask[:, None] & col_mask1[None, :],
                               k_broadcast1.to(tl.float32) * v_block1.to(tl.float32), 0.0)
        acc1 += tl.sum(weighted_v1, axis=0)
    out_ptr0 = output_ptr + base_col
    tl.store(out_ptr0, acc0, mask=col_mask0)
    out_ptr1 = output_ptr + base_col + block_size
    tl.store(out_ptr1, acc1, mask=col_mask1)

def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)
    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)
    blk_size = 128
    block_size = 64
    k_size = k.size(0)
    num_blocks_per_program: tl.constexpr = 2
    grid = (triton.cdiv(v_cols, block_size * num_blocks_per_program),)
    rwkv_mm_sparsity_kernel[grid](
        k,
        v,
        output,
        v_cols,
        blk_size,
        k_size,
        block_size,
    )
    return output

def test_rwkv_mm_sparsity():
    torch.manual_seed(0)
    k_size = 1024
    v_cols = 512
    k = torch.randn(k_size, device='npu', dtype=torch.float32)
    v = torch.randn(k_size, v_cols, device='npu', dtype=torch.float32)
    k_sparse = k.clone()
    k_sparse[::2] = 0
    for i in range(11):
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