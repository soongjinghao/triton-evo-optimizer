import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    K: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)
    num_nope_blocks = NUM_NOPE_BLOCKS
    num_rope_blocks = tl.cdiv(DIM_ROPE, GROUP_SIZE)
    total_nope_programs = tl.cdiv(num_nope_blocks, K)
    if raw_block_id < total_nope_programs:
        base_block_id = raw_block_id * K
        for g in tl.static_range(K):
            block_id = base_block_id + g
            if block_id < num_nope_blocks:
                offs = block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
                mask = offs < DIM_NOPE
                ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs
                y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
                y_s = tl.max(tl.abs(y)) / FP8_MAX
                y_s_inv = 1.0 / y_s
                y_q = tl.clamp(y * y_s_inv, -448.0, 448.0).to(
                    output_nope_q_ptr.dtype.element_ty
                )
                dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
                dst_s_ptr = (
                    output_nope_s_ptr + token_id * output_nope_s_stride_0 + block_id
                )
                tl.store(dst_q_ptr, y_q, mask=mask)
                tl.store(dst_s_ptr, y_s)
    else:
        rope_block_id = raw_block_id - total_nope_programs
        if rope_block_id < num_rope_blocks:
            offs = rope_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            mask = offs < DIM_ROPE
            src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
            dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs
            data = tl.load(src_ptr, mask=mask)
            tl.store(dst_ptr, data, mask=mask)

def _quantize_k_cache_fast(k_nope, k_rope, group_size: int = 128):
    assert k_nope.dtype == torch.bfloat16
    assert k_rope.dtype == torch.bfloat16
    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape
    assert num_tokens == num_tokens_
    assert dim_nope == 512
    assert dim_rope == 64
    assert k_nope.dtype == k_rope.dtype
    num_tiles = dim_nope // group_size
    assert k_nope.stride(1) == 1
    assert k_rope.stride(1) == 1
    output = torch.empty(
        (num_tokens, dim_nope + num_tiles * 4 + dim_rope),
        dtype=torch.bfloat16,
        device=k_nope.device,
    )
    output_nope_q = output[..., :dim_nope]
    output_nope_s = output[..., dim_nope : dim_nope + num_tiles * 4].view(torch.bfloat16)
    output_rope = output[..., dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert dim_nope % group_size == 0
    NUM_NOPE_BLOCKS = dim_nope // group_size
    K = 4
    total_nope_programs = triton.cdiv(NUM_NOPE_BLOCKS, K)
    num_rope_blocks = triton.cdiv(dim_rope, group_size)
    total_programs = total_nope_programs + num_rope_blocks
    grid = lambda meta: (min(num_tokens, 65535), min(total_programs, 65535))
    _quantize_k_cache_fast_kernel[grid](
        output_nope_q,
        output_nope_s,
        output_rope,
        k_nope,
        k_rope,
        output_nope_q.stride(0),
        output_nope_s.stride(0),
        output_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        K=K,
    )
    return output