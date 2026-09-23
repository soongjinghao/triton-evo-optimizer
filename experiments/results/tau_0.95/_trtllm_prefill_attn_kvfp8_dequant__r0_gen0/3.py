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
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    page_group_idx = tl.program_id(1).to(tl.int64)

    page_offsets = tl.arange(0, BLOCK_PAGES).to(tl.int64)
    page_indices = page_group_idx * BLOCK_PAGES + page_offsets

    page_ids = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + page_indices,
        mask=page_indices < block_table_stride,
        other=0,
    ).to(tl.int64)

    valid_pages = (page_indices < block_table_stride) & (page_ids > 0)

    page_ids_2d = tl.view(page_ids, (BLOCK_PAGES, 1))
    page_indices_2d = tl.view(page_indices, (BLOCK_PAGES, 1))
    valid_2d = tl.view(valid_pages, (BLOCK_PAGES, 1))
    cols_2d = tl.view(tl.arange(0, K_CACHE_STRIDE), (1, K_CACHE_STRIDE))

    src_k_offsets = page_ids_2d * KV_CACHE_STRIDE + cols_2d
    dst_k_offsets = (
        batch_idx * block_table_stride + page_indices_2d + 1
    ) * KV_CACHE_STRIDE + cols_2d

    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    fp8_vals_k = tl.load(kv_cache_ptr + src_k_offsets, mask=valid_2d, other=0.0)
    dequantized_k = fp8_vals_k.to(tl.float32) * k_scale_val
    tl.store(mock_kv_cache_ptr + dst_k_offsets, dequantized_k.to(dequant_dtype), mask=valid_2d)

    src_v_offsets = src_k_offsets + K_CACHE_STRIDE
    fp8_vals_v = tl.load(kv_cache_ptr + src_v_offsets, mask=valid_2d, other=0.0)
    dequantized_v = fp8_vals_v.to(tl.float32) * v_scale_val
    dst_v_offsets = dst_k_offsets + K_CACHE_STRIDE
    tl.store(mock_kv_cache_ptr + dst_v_offsets, dequantized_v.to(dequant_dtype), mask=valid_2d)


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

    if k_cache_stride <= 1024:
        BLOCK_PAGES = 8
    elif k_cache_stride <= 4096:
        BLOCK_PAGES = 4
    else:
        BLOCK_PAGES = 2

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)

    grid = (batch_size, (num_of_page_per_token + BLOCK_PAGES - 1) // BLOCK_PAGES)

    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
        BLOCK_PAGES,
    )

    return mock_kv_cache, mock_block_table