from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional
import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1
logger = None


@triton.jit
def _precompute_block_metadata_kernel(
    token_indices_ptr,
    precomputed_block_id_ptr,
    precomputed_inblock_off_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
    pb_stride0,
    pb_stride1,
    po_stride0,
    po_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)
    is_invalid_tok = tok < 0
    tok_safe = tl.where(is_invalid_tok, 0, tok)
    block_id = tok_safe // BLOCK_SIZE
    inblock_off = tok_safe % BLOCK_SIZE
    block_id = tl.where(is_invalid_tok, -1, block_id)
    inblock_off = tl.where(is_invalid_tok, -1, inblock_off)
    pb_ptr = precomputed_block_id_ptr + token_id * pb_stride0 + indice_id * pb_stride1
    po_ptr = precomputed_inblock_off_ptr + token_id * po_stride0 + indice_id * po_stride1
    tl.store(pb_ptr, block_id)
    tl.store(po_ptr, inblock_off)


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    precomputed_block_id_ptr,
    precomputed_inblock_off_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bt_stride0,
    bt_stride1,
    pb_stride0,
    pb_stride1,
    po_stride0,
    po_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    req = tl.load(req_id_ptr + token_id)
    pb_ptr = precomputed_block_id_ptr + token_id * pb_stride0 + indice_id * pb_stride1
    po_ptr = precomputed_inblock_off_ptr + token_id * po_stride0 + indice_id * po_stride1
    block_id = tl.load(pb_ptr)
    inblock_off = tl.load(po_ptr)
    is_invalid_tok = block_id < 0
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)
    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)
    precomputed_block_id = torch.empty_like(token_indices_c)
    precomputed_inblock_off = torch.empty_like(token_indices_c)
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    pb_stride0, pb_stride1 = precomputed_block_id.stride()
    po_stride0, po_stride1 = precomputed_inblock_off.stride()
    out_stride0, out_stride1 = out.stride()
    grid = (num_tokens, tiles_per_row)
    _precompute_block_metadata_kernel[grid](
        token_indices_c,
        precomputed_block_id,
        precomputed_inblock_off,
        BLOCK_SIZE,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        pb_stride0,
        pb_stride1,
        po_stride0,
        po_stride1,
    )
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        precomputed_block_id,
        precomputed_inblock_off,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        bt_stride0,
        bt_stride1,
        pb_stride0,
        pb_stride1,
        po_stride0,
        po_stride1,
        out_stride0,
        out_stride1,
    )
    return out