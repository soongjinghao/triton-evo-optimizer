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
    FP16_ELEMS_PER_PAGE: tl.constexpr,
    FP32_ELEMS_PER_PAGE: tl.constexpr,
    K_STRIDE_FP16: tl.constexpr,
    SCALE_OFFSET_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program processes exactly one token
    loc = tl.load(loc_ptr + pid)
    k_scale = tl.load(index_k_scale_ptr + pid)

    # Load the whole k row for this token
    offs = tl.arange(0, K_STRIDE_FP16)
    off_k = pid * index_k_ptr_stride_0 + offs
    k = tl.load(index_k_ptr + off_k)

    # Integer division and modulo for page index and offset (optimal on NPU)
    page_idx = loc // PAGE_SIZE
    offset_in_page = loc % PAGE_SIZE

    # Store k (fp16)
    out_k_offsets = (
        page_idx * FP16_ELEMS_PER_PAGE
        + offset_in_page * K_STRIDE_FP16
        + offs
    )
    tl.store(buf_fp16_ptr + out_k_offsets, k)

    # Store scale (fp32)
    out_s_offset = (
        page_idx * FP32_ELEMS_PER_PAGE
        + SCALE_OFFSET_FP32
        + offset_in_page
    )
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    :param buf: (num_pages, page_size * (index_head_dim*2 + 4)), uint8
    :param loc: (num_tokens_to_write,), int64, the token index to write to
    :param index_k: (num_tokens_to_write, index_head_dim), fp16
    :param index_k_scale: (num_tokens_to_write,) or (num_tokens_to_write, 1), fp32
    """
    num_pages, buf_numel_per_page = buf.shape
    num_tokens_to_write, index_head_dim = index_k.shape

    # Normalize index_k_scale to 1D
    if index_k_scale.ndim == 2:
        index_k_scale = index_k_scale.reshape(-1)
    assert index_k_scale.shape[0] == num_tokens_to_write

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert index_k.dtype == torch.float16
    assert index_k_scale.dtype == torch.float32

    # Compute layout constants dynamically
    fp16_elems_per_page = page_size * index_head_dim
    fp32_elems_per_page = page_size * (index_head_dim // 2 + 1)
    k_stride_fp16 = index_head_dim
    scale_offset_fp32 = page_size * index_head_dim // 2

    grid = (num_tokens_to_write,)

    buf_fp16 = buf.view(torch.float16)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_triton_kernel[grid](
        buf_fp16,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        FP16_ELEMS_PER_PAGE=fp16_elems_per_page,
        FP32_ELEMS_PER_PAGE=fp32_elems_per_page,
        K_STRIDE_FP16=k_stride_fp16,
        SCALE_OFFSET_FP32=scale_offset_fp32,
    )