import torch
import triton
from vllm.platforms import current_platform

from reduce_segments import reduce_segments

# Import the Triton kernel function (assuming it's in a module)
# For this test, we'll define a wrapper to call the kernel

def call_reduce_segments(
    output, segm_output, segm_max, segm_expsum, seq_lens, query_start_len,
    num_query_heads, out_scale_inv, HEAD_SIZE, HEAD_SIZE_PADDED, 
    TILE_SIZE, NUM_SEGMENTS_PER_SEQ, USE_FP8
):
    num_seqs = seq_lens.shape[0]
    num_tokens = output.shape[0]
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    block_table_stride = 0  # Not used in this kernel
    BLOCK_Q = 1  # Not used in this kernel
    
    grid = (num_tokens, num_query_heads)
    
    reduce_segments[grid](
        output_ptr=output,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seq_lens,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        out_scale_inv=out_scale_inv,
        output_stride_0=output_stride_0,
        output_stride_1=output_stride_1,
        block_table_stride=block_table_stride,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        query_start_len_ptr=query_start_len,
        BLOCK_Q=BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS_PER_SEQ,
        USE_FP8=USE_FP8,
        num_warps=4,
        num_stages=2
    )

if __name__ == "__main__":
    # Test parameters
    device = torch.device("npu")
    
    # Test case 1: Basic functionality
    num_tokens = 4
    num_query_heads = 2
    HEAD_SIZE = 8
    HEAD_SIZE_PADDED = 8
    TILE_SIZE = 4
    NUM_SEGMENTS_PER_SEQ = 3
    USE_FP8 = False
    
    # Create input tensors
    output = torch.zeros((num_tokens, num_query_heads, HEAD_SIZE), dtype=torch.float32, device=device)
    segm_output = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED), dtype=torch.float32, device=device)
    segm_max = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ), dtype=torch.float32, device=device)
    segm_expsum = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ), dtype=torch.float32, device=device)
    seq_lens = torch.tensor([8, 4], dtype=torch.int32, device=device)
    query_start_len = torch.tensor([0, 8, 12], dtype=torch.int32, device=device)  # Cumulative sequence lengths
    out_scale_inv = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    # Call the kernel
    call_reduce_segments(
        output, segm_output, segm_max, segm_expsum, seq_lens, query_start_len,
        num_query_heads, out_scale_inv, HEAD_SIZE, HEAD_SIZE_PADDED, 
        TILE_SIZE, NUM_SEGMENTS_PER_SEQ, USE_FP8
    )
    
    # Reference implementation
    def reference_reduce_segments(
        segm_output, segm_max, segm_expsum, seq_lens, query_start_len,
        num_query_heads, HEAD_SIZE
    ):
        num_tokens = segm_output.shape[0]
        output = torch.zeros((num_tokens, num_query_heads, HEAD_SIZE), dtype=torch.float32)
        
        # Determine sequence indices for each token
        seq_indices = torch.zeros(num_tokens, dtype=torch.int32)
        for i in range(num_tokens):
            for j in range(len(query_start_len) - 1):
                if query_start_len[j] <= i < query_start_len[j+1]:
                    seq_indices[i] = j
                    break
        
        for token_idx in range(num_tokens):
            seq_idx = seq_indices[token_idx].item()
            seq_len = seq_lens[seq_idx].item()
            
            # Calculate actual number of segments
            tiles_per_segment = (seq_len + NUM_SEGMENTS_PER_SEQ * TILE_SIZE - 1) // (NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
            act_num_segments = (seq_len + tiles_per_segment * TILE_SIZE - 1) // (tiles_per_segment * TILE_SIZE)
            
            # Get valid segments
            valid_segm_max = segm_max[token_idx, :, :act_num_segments]
            valid_segm_expsum = segm_expsum[token_idx, :, :act_num_segments]
            valid_segm_output = segm_output[token_idx, :, :act_num_segments, :HEAD_SIZE]
            
            # Compute overall max
            overall_max = torch.max(valid_segm_max, dim=1, keepdim=True).values
            
            # Rescale expsums
            rescaled_expsum = valid_segm_expsum * torch.exp(valid_segm_max - overall_max)
            overall_expsum = torch.sum(rescaled_expsum, dim=1, keepdim=True)
            
            # Rescale outputs
            rescaled_output = valid_segm_output * torch.exp(valid_segm_max - overall_max).unsqueeze(-1)
            acc_sum = torch.sum(rescaled_output, dim=1)
            
            # Final result
            result = torch.where(overall_expsum == 0, torch.zeros_like(acc_sum), acc_sum / overall_expsum)
            output[token_idx] = result
        
        return output
    
    # Compute reference output
    ref_output = reference_reduce_segments(
        segm_output.cpu(), segm_max.cpu(), segm_expsum.cpu(), 
        seq_lens.cpu(), query_start_len.cpu(), num_query_heads, HEAD_SIZE
    ).to(device)
    
    # Validate correctness
    assert torch.allclose(output, ref_output, rtol=1e-4, atol=1e-4), "Output does not match reference implementation"
    print("Basic test passed!")
    
    # Test case 2: Different parameters
    num_tokens = 8
    num_query_heads = 4
    HEAD_SIZE = 16
    HEAD_SIZE_PADDED = 16
    TILE_SIZE = 8
    NUM_SEGMENTS_PER_SEQ = 2
    USE_FP8 = False
    
    output = torch.zeros((num_tokens, num_query_heads, HEAD_SIZE), dtype=torch.float32, device=device)
    segm_output = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED), dtype=torch.float32, device=device)
    segm_max = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ), dtype=torch.float32, device=device)
    segm_expsum = torch.randn((num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ), dtype=torch.float32, device=device)
    seq_lens = torch.tensor([16, 8, 4, 4], dtype=torch.int32, device=device)
    query_start_len = torch.tensor([0, 16, 24, 28, 32], dtype=torch.int32, device=device)
    out_scale_inv = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    call_reduce_segments(
        output, segm_output, segm_max, segm_expsum, seq_lens, query_start_len,
        num_query_heads, out_scale_inv, HEAD_SIZE, HEAD_SIZE_PADDED, 
        TILE_SIZE, NUM_SEGMENTS_PER_SEQ, USE_FP8
    )
    
    ref_output = reference_reduce_segments(
        segm_output.cpu(), segm_max.cpu(), segm_expsum.cpu(), 
        seq_lens.cpu(), query_start_len.cpu(), num_query_heads, HEAD_SIZE
    ).to(device)
    
    assert torch.allclose(output, ref_output, rtol=1e-4, atol=1e-4), "Output does not match reference implementation for test case 2"
    print("Test case 2 passed!")
    
    print("All tests passed!")