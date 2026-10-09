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
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 : dim_nope + num_tiles * 4 + dim_rope].view(torch.bfloat16)
    _dequantize_k_cache_fast_kernel[(triton.cdiv(num_tokens, 1), num_blocks_per_token)](
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
    raw_block_id = tl.program_id(1)
    
    # Hoist token_id * stride computations as loop invariants
    token_offset_output = token_id * output_stride_0
    token_offset_nope_q = token_id * input_nope_q_stride_0
    token_offset_nope_s = token_id * input_nope_s_stride_0
    token_offset_rope = token_id * input_rope_stride_0
    
    # Unified loop over all blocks (both nope and rope)
    offs_base = raw_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    
    # Determine if this block is in nope or rope region
    is_nope_block = raw_block_id < NUM_NOPE_BLOCKS
    
    # Compute effective block id and dimension bounds
    effective_block_id = tl.where(is_nope_block, raw_block_id, raw_block_id - NUM_NOPE_BLOCKS)
    dim_bound = tl.where(is_nope_block, DIM_NOPE, DIM_ROPE)
    
    # Compute offsets within the region
    offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < dim_bound
    
    # Load and compute for nope blocks
    ptr_q = input_nope_q_ptr + token_offset_nope_q + offs
    ptr_s = input_nope_s_ptr + token_offset_nope_s + effective_block_id
    y_q = tl.load(ptr_q, mask=mask & is_nope_block, other=0.0).to(tl.float32)
    y_s = tl.load(ptr_s, mask=is_nope_block, other=0.0)
    y_nope = (y_q * y_s).to(output_ptr.dtype.element_ty)
    
    # Load for rope blocks
    src_ptr = input_rope_ptr + token_offset_rope + offs
    data_rope = tl.load(src_ptr, mask=mask & ~is_nope_block, other=0.0).to(tl.bfloat16)
    
    # Select output based on block type
    y = tl.where(is_nope_block, y_nope, data_rope)
    
    # Compute output offset: nope goes to [0, DIM_NOPE), rope goes to [DIM_NOPE, DIM_NOPE+DIM_ROPE)
    output_offset = tl.where(is_nope_block, offs, DIM_NOPE + offs)
    dst_ptr = output_ptr + token_offset_output + output_offset
    tl.store(dst_ptr, y, mask=mask)