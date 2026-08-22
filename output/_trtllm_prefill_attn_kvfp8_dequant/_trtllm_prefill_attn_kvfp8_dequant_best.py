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
FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8
trtllm_gen_workspace_buffer = None
@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    mock_kv_cache_ptr,
    kv_orig_page_nums_ptr,
    mock_cache_base_offsets_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    orig_page_num = tl.load(kv_orig_page_nums_ptr + pid).to(tl.int64)
    mock_base = tl.load(mock_cache_base_offsets_ptr + pid).to(tl.int64)
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    offset = tl.arange(0, K_CACHE_STRIDE)
    offset_k = orig_page_num * KV_CACHE_STRIDE + offset
    fp8_vals_k = tl.load(tl.multiple_of(kv_cache_ptr + offset_k, 16))
    dequantized_k = fp8_vals_k.to(tl.float32) * k_scale_val
    tl.store(tl.multiple_of(mock_kv_cache_ptr + mock_base + offset, 16), dequantized_k.to(dequant_dtype))
    v_base = mock_base + K_CACHE_STRIDE
    offset_v = orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + offset
    fp8_vals_v = tl.load(tl.multiple_of(kv_cache_ptr + offset_v, 16))
    dequantized_v = fp8_vals_v.to(tl.float32) * v_scale_val
    tl.store(tl.multiple_of(mock_kv_cache_ptr + v_base + offset, 16), dequantized_v.to(dequant_dtype))
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
    valid_mask = block_tables_prefill > 0
    num_valid = valid_mask.sum().item()
    if num_valid == 0:
        return mock_kv_cache, mock_block_table
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)
    batch_idx_valid = valid_indices[:, 0].to(torch.int64)
    mock_block_idx_valid = valid_indices[:, 1].to(torch.int64)
    orig_page_nums = block_tables_prefill[valid_mask].to(torch.int64)
    mock_base_offsets = (batch_idx_valid * num_of_page_per_token + mock_block_idx_valid + 1) * kv_cache_stride
    mock_base_offsets = mock_base_offsets.to(torch.int64)
    grid = (num_valid,)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        mock_kv_cache,
        orig_page_nums,
        mock_base_offsets,
        k_scale,
        v_scale,
        K_CACHE_STRIDE=k_cache_stride,
        KV_CACHE_STRIDE=kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table