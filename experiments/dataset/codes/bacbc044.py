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
    orig_page_num_ptr,
    valid_mask_ptr,
):
    pid = tl.program_id(0).to(tl.int64)
    valid = tl.load(valid_mask_ptr + pid).to(tl.int1)
    if not valid:
        return
    orig_page_num = tl.load(orig_page_num_ptr + pid).to(tl.int64)
    batch_idx = pid // block_table_stride
    mock_block_table_idx = pid % block_table_stride
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    k_scale_val = tl.load(k_scale_ptr)
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val
    base_offset = (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + base_offset + tl.arange(0, K_CACHE_STRIDE), dequantized_vals)
    v_scale_val = tl.load(v_scale_ptr)
    offset = (
        orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    )
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + base_offset + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE), dequantized_vals)
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
    total_programs = batch_size * num_of_page_per_token
    orig_page_num = block_tables_prefill.reshape(-1).to(torch.int64)
    valid_mask = (orig_page_num > 0).to(torch.int32)
    grid = (total_programs,)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
        orig_page_num,
        valid_mask,
    )
    return mock_kv_cache, mock_block_table