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
    BLOCK_M = 4
    BLOCK_GROUPS = 4
    num_token_blocks = triton.cdiv(num_tokens, BLOCK_M)
    num_group_blocks = triton.cdiv(num_blocks_per_token, BLOCK_GROUPS)
    grid = (num_token_blocks * num_group_blocks,)
    _dequantize_k_cache_fast_kernel[grid](
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
        BLOCK_M=BLOCK_M,
        BLOCK_GROUPS=BLOCK_GROUPS,
        NUM_TOKENS=num_tokens,
        NUM_BLOCKS_PER_TOKEN=num_blocks_per_token,
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
    BLOCK_M: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_BLOCKS_PER_TOKEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_group_blocks = tl.cdiv(NUM_BLOCKS_PER_TOKEN, BLOCK_GROUPS)
    token_block_id = pid // num_group_blocks
    group_block_id = pid % num_group_blocks
    
    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = offs_token < NUM_TOKENS
    
    offs_group = group_block_id * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    group_mask = offs_group < NUM_BLOCKS_PER_TOKEN
    
    is_nope = offs_group < NUM_NOPE_BLOCKS
    is_rope = (offs_group >= NUM_NOPE_BLOCKS) & (offs_group < NUM_BLOCKS_PER_TOKEN)
    
    offs_col = offs_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    col_mask = offs_col < DIM_NOPE + DIM_ROPE
    
    full_mask = token_mask[:, None] & group_mask[None, :] & col_mask[None, None, :]
    
    nope_col_mask = offs_col < DIM_NOPE
    rope_col_mask = (offs_col >= DIM_NOPE) & (offs_col < DIM_NOPE + DIM_ROPE)
    
    nope_mask = full_mask & is_nope[None, :, None] & nope_col_mask[None, None, :]
    rope_mask = full_mask & is_rope[None, :, None] & rope_col_mask[None, None, :]
    
    ptr_q = input_nope_q_ptr + offs_token[:, None, None] * input_nope_q_stride_0 + offs_col[None, None, :]
    ptr_s = input_nope_s_ptr + offs_token[:, None, None] * input_nope_s_stride_0 + offs_group[None, :, None]
    ptr_rope = input_rope_ptr + offs_token[:, None, None] * input_rope_stride_0 + (offs_col - DIM_NOPE)[None, None, :]
    
    y_q = tl.load(ptr_q, mask=nope_mask, other=0.0).to(tl.float32)
    y_s = tl.load(ptr_s, mask=(token_mask[:, None, None] & group_mask[None, :, None] & is_nope[None, :, None]), other=0.0)
    y_nope = (y_q * y_s).to(output_ptr.dtype.element_ty)
    
    y_rope = tl.load(ptr_rope, mask=rope_mask, other=0.0).to(tl.bfloat16)
    
    dst_ptr = output_ptr + offs_token[:, None, None] * output_stride_0 + offs_col[None, None, :]
    
    tl.store(dst_ptr, tl.where(is_nope[None, :, None], y_nope, y_rope), mask=full_mask)