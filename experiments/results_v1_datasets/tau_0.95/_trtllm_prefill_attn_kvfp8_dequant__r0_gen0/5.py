from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import torch
import logging
import triton
import triton.language as tl

PAD_SLOT_ID = -1
logger = logging.getLogger(__name__)
FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
FP8_DTYPE = torch.float8_e4m3fn  # NPU FP8 dtype
FP4_DTYPE = torch.uint8
trtllm_gen_workspace_buffer = None


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    num_pages,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    page_block_idx = tl.program_id(1).to(tl.int64)

    local_pages = tl.arange(0, BLOCK_PAGES)
    page_indices = page_block_idx * BLOCK_PAGES + local_pages
    page_mask = page_indices < num_pages

    orig_page_nums = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + page_indices,
        mask=page_mask,
        other=0,
    ).to(tl.int64)

    valid_page_mask = page_mask & (orig_page_nums > 0)

    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    col_offsets = tl.arange(0, 2 * K_CACHE_STRIDE)
    load_base = orig_page_nums * KV_CACHE_STRIDE
    load_ptrs = kv_cache_ptr + load_base[:, None] + col_offsets[None, :]
    valid_page_mask_2d = valid_page_mask[:, None]
    fp8_vals_2d = tl.load(load_ptrs, mask=valid_page_mask_2d, other=0.0)

    col_scales = tl.where(
        col_offsets < K_CACHE_STRIDE,
        k_scale_val,
        v_scale_val,
    )
    dequantized_2d = fp8_vals_2d.to(tl.float32) * col_scales[None, :].to(tl.float32)

    out_base = (batch_idx * block_table_stride + page_indices + 1) * KV_CACHE_STRIDE
    out_ptrs = mock_kv_cache_ptr + out_base[:, None] + col_offsets[None, :]
    tl.store(out_ptrs, dequantized_2d.to(dequant_dtype), mask=valid_page_mask_2d)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv_cache.device.type == 'npu', "kv_cache must be on NPU"
    assert block_tables_prefill.device.type == 'npu', "block_tables_prefill must be on NPU"
    assert k_scale.device.type == 'npu', "k_scale must be on NPU"
    assert v_scale.device.type == 'npu', "v_scale must be on NPU"

    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)

    # Shape-aware page packing: avoid oversized on-chip working sets while still
    # merging multiple pages per program.
    max_elems_per_program = 65536
    block_pages = 1
    if num_of_page_per_token >= 8 and (8 * 2 * k_cache_stride) <= max_elems_per_program:
        block_pages = 8
    elif num_of_page_per_token >= 4 and (4 * 2 * k_cache_stride) <= max_elems_per_program:
        block_pages = 4
    elif num_of_page_per_token >= 2 and (2 * 2 * k_cache_stride) <= max_elems_per_program:
        block_pages = 2
    else:
        block_pages = 1

    grid = (batch_size, triton.cdiv(num_of_page_per_token, block_pages))
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
        block_pages,
    )

    return mock_kv_cache, mock_block_table