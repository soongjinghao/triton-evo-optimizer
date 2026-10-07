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
    _dequantize_k_cache_fast_kernel[(triton.cdiv(num_tokens, 1), num_blocks_per_token)](
        output,
        quant_k_cache,
        output.stride(0),
        quant_k_cache.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )
    return output

@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    quant_k_cache_ptr,
    output_stride_0: int,
    quant_k_cache_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)
    offs = raw_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < DIM_NOPE + DIM_ROPE
    src_ptr = quant_k_cache_ptr + token_id * quant_k_cache_stride_0 + offs
    data = tl.load(src_ptr, mask=mask, other=0.0)
    scale_block_id = raw_block_id
    scale_offs = scale_block_id
    scale_ptr = quant_k_cache_ptr + token_id * quant_k_cache_stride_0 + DIM_NOPE + scale_offs
    scale = tl.load(scale_ptr)
    scale = scale.to(tl.float32)
    data_f32 = data.to(tl.float32)
    nope_mask = offs < DIM_NOPE
    result = tl.where(nope_mask, data_f32 * scale, data_f32)
    result = result.to(output_ptr.dtype.element_ty)
    dst_ptr = output_ptr + token_id * output_stride_0 + offs
    tl.store(dst_ptr, result, mask=mask)