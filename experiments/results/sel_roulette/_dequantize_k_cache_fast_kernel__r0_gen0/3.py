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
    _dequantize_k_cache_fast_kernel[(triton.cdiv(num_tokens, 1), 1)](
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
    DIM_TOTAL: tl.constexpr = DIM_NOPE + DIM_ROPE
    offs = tl.arange(0, DIM_TOTAL)
    mask_nope = offs < DIM_NOPE
    mask_rope = offs >= DIM_NOPE
    offs_nope = offs
    offs_rope = offs - DIM_NOPE
    block_id_nope = offs_nope // GROUP_SIZE
    offs_in_block = offs_nope % GROUP_SIZE
    ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_nope
    ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + block_id_nope
    y_q = tl.load(ptr_q, mask=mask_nope, other=0.0).to(tl.float32)
    y_s = tl.load(ptr_s, mask=mask_nope, other=0.0)
    y_nope = (y_q * y_s).to(output_ptr.dtype.element_ty)
    src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs_rope
    y_rope = tl.load(src_ptr, mask=mask_rope, other=0.0).to(tl.bfloat16)
    dst_ptr = output_ptr + token_id * output_stride_0 + offs
    tl.store(dst_ptr, tl.where(mask_nope, y_nope, y_rope), mask=(offs < DIM_TOTAL))