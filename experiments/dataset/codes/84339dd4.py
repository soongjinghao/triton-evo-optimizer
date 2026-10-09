import torch
import triton
import triton.language as tl
device = 'npu'
@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp16_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    pid = tl.program_id(0)
    tokens_per_program = 4
    token_id_base = pid * tokens_per_program
    offsets = token_id_base + tl.arange(0, tokens_per_program)
    loc = tl.load(loc_ptr + offsets, mask=offsets < tl.num_programs(0) * tokens_per_program, other=0)
    in_k_offsets = offsets[:, None] * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)[None, :]
    k = tl.load(index_k_ptr + in_k_offsets, mask=offsets[:, None] < tl.num_programs(0) * tokens_per_program)
    k_scale = tl.load(index_k_scale_ptr + offsets, mask=offsets < tl.num_programs(0) * tokens_per_program, other=0.0)
    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE
    page_base_k = loc_page_index * BUF_NUMEL_PER_PAGE
    token_offset_k = loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
    out_k_offsets = page_base_k[:, None] + token_offset_k[:, None] + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)[None, :]
    page_base_s = loc_page_index * (BUF_NUMEL_PER_PAGE // 4)
    token_offset_s = loc_token_offset_in_page + (S_OFFSET_NBYTES_IN_PAGE // 4)
    out_s_offset = page_base_s + token_offset_s
    tl.store(buf_fp16_ptr + out_k_offsets, k, mask=offsets[:, None] < tl.num_programs(0) * tokens_per_program)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale, mask=offsets < tl.num_programs(0) * tokens_per_program)
def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )
    assert buf_numel_per_page == 64 * (128 + 4)
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    assert page_size == 64
    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"
    assert index_k.dtype == torch.float16
    assert index_k_scale.dtype == torch.float32
    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()
    buf_fp16 = buf.view(torch.float16)
    buf_fp32 = buf.view(torch.float32)
    tokens_per_program = 4
    grid = (num_tokens_to_write + tokens_per_program - 1) // tokens_per_program
    _set_k_and_s_triton_kernel[(grid,)](
        buf_fp16,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )