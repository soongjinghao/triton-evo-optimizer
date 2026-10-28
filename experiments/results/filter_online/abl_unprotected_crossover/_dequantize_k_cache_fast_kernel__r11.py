import torch
import triton
import triton.language as tl

def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128):
    num_tokens, dim_quant = quant_k_cache.shape
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    assert dim_quant == 656
    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5
    assert dim_nope % group_size == 0
    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    _dequantize_k_cache_fast_kernel[(num_tokens, 1)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )
    return output

@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    K: tl.constexpr = 5
    TOTAL_GROUP_SIZE: tl.constexpr = GROUP_SIZE * K
    offs_total = tl.arange(0, TOTAL_GROUP_SIZE)
    
    # Process 5 blocks: first NUM_NOPE_BLOCKS (4) are nope quant, last 1 is rope
    # Load all 640 elements from nope_q (first 512) and rope (next 128) in one contiguous load
    # Since input_nope_q and input_rope are physically contiguous in memory (dim_quant=656, dim_nope=512, dim_rope=64)
    # We load from input_nope_q base for first 512, then from input_rope base for next 128
    # But they are separate pointers, so we load separately
    
    # Load nope quantized data (4 blocks * 128 = 512 elements)
    offs_q = tl.arange(0, GROUP_SIZE * NUM_NOPE_BLOCKS)
    mask_q = offs_q < DIM_NOPE
    ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
    y_q = tl.load(ptr_q, mask=mask_q, other=0.0).to(tl.float32)
    
    # Load scales (4 scales)
    offs_s = tl.arange(0, NUM_NOPE_BLOCKS)
    ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + offs_s
    y_s = tl.load(ptr_s)
    
    # Reshape y_q to (NUM_NOPE_BLOCKS, GROUP_SIZE) and multiply by scales
    y_q_2d = tl.view(y_q, (NUM_NOPE_BLOCKS, GROUP_SIZE))
    y_q_scaled = y_q_2d * y_s[:, None]
    
    # Reshape back to 1D and store
    y_nope = tl.view(y_q_scaled, (DIM_NOPE,))
    dst_ptr_nope = output_ptr + token_id * output_stride_0 + offs_q
    tl.store(dst_ptr_nope, y_nope.to(output_ptr.dtype.element_ty), mask=mask_q)
    
    # Load rope data (1 block * 128 = 128 elements)
    offs_rope = tl.arange(0, GROUP_SIZE)
    mask_rope = offs_rope < DIM_ROPE
    src_ptr_rope = input_rope_ptr + token_id * input_rope_stride_0 + offs_rope
    dst_ptr_rope = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs_rope
    data_rope = tl.load(src_ptr_rope, mask=mask_rope).to(tl.bfloat16)
    tl.store(dst_ptr_rope, data_rope, mask=mask_rope)