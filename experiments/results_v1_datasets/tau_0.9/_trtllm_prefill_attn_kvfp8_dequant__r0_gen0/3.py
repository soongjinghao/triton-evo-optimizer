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
):
    # 标量阶段:集中计算所有标量地址与索引
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    logical_row = batch_idx * block_table_stride + mock_block_table_idx

    orig_page_num = tl.load(
        block_tables_prefill_ptr + logical_row
    ).to(tl.int64)

    if orig_page_num <= 0:
        return

    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    rows = tl.arange(0, K_CACHE_STRIDE)
    src_base = orig_page_num * KV_CACHE_STRIDE
    dst_base = (logical_row + 1) * KV_CACHE_STRIDE

    k_src_offsets = src_base + rows
    v_src_offsets = src_base + K_CACHE_STRIDE + rows
    k_dst_offsets = dst_base + rows
    v_dst_offsets = dst_base + K_CACHE_STRIDE + rows

    # 连续发起标量 scale 加载
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    # 连续发起向量 fp8 加载
    k_fp8_vals = tl.load(kv_cache_ptr + k_src_offsets)
    v_fp8_vals = tl.load(kv_cache_ptr + v_src_offsets)

    # 统一执行 dtype 转换与 scale 乘法
    k_dequantized_vals = (k_fp8_vals.to(tl.float32) * k_scale_val).to(dequant_dtype)
    v_dequantized_vals = (v_fp8_vals.to(tl.float32) * v_scale_val).to(dequant_dtype)

    # 集中写回 mock KV cache
    tl.store(mock_kv_cache_ptr + k_dst_offsets, k_dequantized_vals)
    tl.store(mock_kv_cache_ptr + v_dst_offsets, v_dequantized_vals)


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
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table