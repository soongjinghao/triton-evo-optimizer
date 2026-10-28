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
    BLOCK_TOKENS = 4
    assert num_tokens % BLOCK_TOKENS == 0
    grid = (num_tokens // BLOCK_TOKENS, 5)
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
        BLOCK_TOKENS=BLOCK_TOKENS,
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
    BLOCK_TOKENS: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)
    base_token = token_block_id * BLOCK_TOKENS
    offs_tokens = base_token + tl.arange(0, BLOCK_TOKENS)
    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask_q = offs_q < DIM_NOPE
        offs_2d = offs_tokens[:, None] * input_nope_q_stride_0 + offs_q[None, :]
        ptr_q = input_nope_q_ptr + offs_2d
        y_q = tl.load(ptr_q, mask=mask_q[None, :], other=0.0).to(tl.float32)
        offs_s = offs_tokens * input_nope_s_stride_0 + effective_block_id
        ptr_s = input_nope_s_ptr + offs_s
        y_s = tl.load(ptr_s).to(tl.float32)
        y = (y_q * y_s[:, None]).to(output_ptr.dtype.element_ty)
        offs_dst = offs_tokens[:, None] * output_stride_0 + offs_q[None, :]
        dst_ptr = output_ptr + offs_dst
        tl.store(dst_ptr, y, mask=mask_q[None, :])
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        offs_2d = offs_tokens[:, None] * input_rope_stride_0 + offs[None, :]
        src_ptr = input_rope_ptr + offs_2d
        data = tl.load(src_ptr, mask=mask[None, :]).to(tl.bfloat16)
        offs_dst = offs_tokens[:, None] * output_stride_0 + DIM_NOPE + offs[None, :]
        dst_ptr = output_ptr + offs_dst
        tl.store(dst_ptr, data, mask=mask[None, :])