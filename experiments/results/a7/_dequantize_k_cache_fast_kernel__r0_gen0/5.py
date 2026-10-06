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
    sm_q = tl.zeros([DIM_NOPE], dtype=tl.int8)
    sm_s = tl.zeros([NUM_NOPE_BLOCKS], dtype=tl.float32)
    for i in range(NUM_NOPE_BLOCKS):
        offs_q = i * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
        sm_q = tl.where(mask, tl.load(ptr_q, mask=mask, other=0), sm_q)
        ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + i
        sm_s = tl.where(i == tl.arange(0, NUM_NOPE_BLOCKS), tl.load(ptr_s), sm_s)
    for i in range(NUM_NOPE_BLOCKS):
        offs_q = i * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        y = sm_q[offs_q].to(tl.float32) * sm_s[i]
        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y.to(tl.bfloat16), mask=mask)
    offs_rope = tl.arange(0, GROUP_SIZE)
    mask_rope = offs_rope < DIM_ROPE
    src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs_rope
    dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs_rope
    data = tl.load(src_ptr, mask=mask_rope).to(tl.bfloat16)
    tl.store(dst_ptr, data, mask=mask_rope)