from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional
import numpy as np
import torch
import triton
import triton.language as tl
PAD_SLOT_ID = -1
logger = None  # Simplified logging
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N_PACKED + tl.arange(0, BLOCK_N_PACKED)
    req = tl.load(req_id_ptr + token_id)
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32
    is_invalid_tok = tok < 0
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)
    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)
def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    BLOCK_N_PACKED = BLOCK_N * 4
    assert NUM_TOPK_TOKENS % BLOCK_N_PACKED == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N_PACKED ({BLOCK_N_PACKED})"
    )
    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N_PACKED
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)
    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()
    grid = (num_tokens, tiles_per_row)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N_PACKED,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out