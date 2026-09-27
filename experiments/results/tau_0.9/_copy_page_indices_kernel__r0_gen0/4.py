from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx
    num_full_blocks = num_blocks // BLOCK_SIZE
    tail_size = num_blocks % BLOCK_SIZE
    
    offset = tl.arange(0, BLOCK_SIZE)
    
    # Prefetch first block
    i = 0
    if num_full_blocks > 0:
        block_ids_next = tl.load(row_ptr + i * BLOCK_SIZE + offset, mask=offset < BLOCK_SIZE)
    
    for i in tl.range(0, num_full_blocks - 1):
        block_ids_curr = block_ids_next
        block_ids_next = tl.load(row_ptr + (i + 1) * BLOCK_SIZE + offset, mask=offset < BLOCK_SIZE)
        tl.store(
            page_indices + start_idx + i * BLOCK_SIZE + offset,
            block_ids_curr,
            mask=offset < BLOCK_SIZE,
        )
    
    # Handle last full block
    if num_full_blocks > 0:
        i_last = num_full_blocks - 1
        tl.store(
            page_indices + start_idx + i_last * BLOCK_SIZE + offset,
            block_ids_next,
            mask=offset < BLOCK_SIZE,
        )
    
    # Handle tail block
    if tail_size > 0:
        i_tail = num_full_blocks * BLOCK_SIZE
        tail_mask = offset < tail_size
        block_ids_tail = tl.load(row_ptr + i_tail + offset, mask=tail_mask)
        tl.store(
            page_indices + start_idx + i_tail + offset,
            block_ids_tail,
            mask=tail_mask,
        )