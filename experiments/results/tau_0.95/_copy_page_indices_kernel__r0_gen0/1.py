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
    offset = tl.arange(0, BLOCK_SIZE)
    if num_blocks < BLOCK_SIZE:
        block_ids = tl.load(row_ptr + offset, mask=offset < num_blocks)
        tl.store(
            page_indices + start_idx + offset,
            block_ids,
            mask=offset < num_blocks,
        )
    else:
        for i in tl.range(0, num_blocks, BLOCK_SIZE):
            mask = i + offset < num_blocks
            block_ids = tl.load(row_ptr + i + offset, mask=mask)
            tl.store(
                page_indices + start_idx + i + offset,
                block_ids,
                mask=mask,
            )