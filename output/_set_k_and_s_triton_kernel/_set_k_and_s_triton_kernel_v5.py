import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}, num_stages=1),
        triton.Config({'num_warps': 2}, num_stages=1),
        triton.Config({'num_warps': 4}, num_stages=1),
        triton.Config({'num_warps': 8}, num_stages=1),
        triton.Config({'num_warps': 1}, num_stages=2),
        triton.Config({'num_warps': 2}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
    ],
    key=['num_tokens_to_write'],
)
@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    stride_k0,
    num_tokens_to_write,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(loc_ptr + token_id)
    in_k_offsets = token_id * stride_k0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)
    loc_page_index = loc >> 6
    loc_token_offset_in_page = loc & 63
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )
    out_s_offset = (
        loc_page_index * (BUF_NUMEL_PER_PAGE // 4)
        + (S_OFFSET_NBYTES_IN_PAGE // 4)
        + loc_token_offset_in_page
    )
    tl.store(tl.multiple_of(buf_fp8_ptr + out_k_offsets, 16), k)
    tl.store(tl.multiple_of(buf_fp32_ptr + out_s_offset, 4), k_scale)


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
    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp16,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        num_tokens_to_write,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )