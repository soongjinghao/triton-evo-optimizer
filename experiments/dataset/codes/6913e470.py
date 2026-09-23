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
    total_pages,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    page_idx = pid * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES).to(tl.int64)

    valid = page_idx < total_pages
    batch_idx = page_idx // block_table_stride
    mock_block_table_idx = page_idx % block_table_stride

    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx,
        mask=valid,
        other=0,
    ).to(tl.int64)

    row_valid = valid & (orig_page_num > 0)

    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    cols_k = tl.arange(0, K_CACHE_STRIDE)

    offset_k = orig_page_num[:, None] * KV_CACHE_STRIDE + cols_k[None, :]
    fp8_vals_k = tl.load(
        tl.multiple_of(kv_cache_ptr + offset_k, 16),
        mask=row_valid[:, None],
        other=0,
    )
    dequantized_k = fp8_vals_k.to(tl.float32) * k_scale_val

    mock_cache_offset_k = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1)[:, None] * KV_CACHE_STRIDE
        + cols_k[None, :]
    )
    tl.store(
        tl.multiple_of(mock_kv_cache_ptr + mock_cache_offset_k, 16),
        dequantized_k.to(dequant_dtype),
        mask=row_valid[:, None],
    )

    offset_v = orig_page_num[:, None] * KV_CACHE_STRIDE + K_CACHE_STRIDE + cols_k[None, :]
    fp8_vals_v = tl.load(
        tl.multiple_of(kv_cache_ptr + offset_v, 16),
        mask=row_valid[:, None],
        other=0,
    )
    dequantized_v = fp8_vals_v.to(tl.float32) * v_scale_val

    mock_cache_offset_v = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1)[:, None] * KV_CACHE_STRIDE
        + K_CACHE_STRIDE
        + cols_k[None, :]
    )
    tl.store(
        tl.multiple_of(mock_kv_cache_ptr + mock_cache_offset_v, 16),
        dequantized_v.to(dequant_dtype),
        mask=row_valid[:, None],
    )

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
    total_pages = batch_size * num_of_page_per_token
    BLOCK_PAGES = 4

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)

    grid = (triton.cdiv(total_pages, BLOCK_PAGES),)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        total_pages,
        k_cache_stride,
        kv_cache_stride,
        BLOCK_PAGES=BLOCK_PAGES,
    )

    return mock_kv_cache, mock_block_table