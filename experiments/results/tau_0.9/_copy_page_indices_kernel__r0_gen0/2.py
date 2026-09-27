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
    cu_num_blocks_start,
    cu_num_blocks_end,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks_start + req_idx)
    end_idx = tl.load(cu_num_blocks_end + req_idx)
    num_blocks = end_idx - start_idx
    offset = tl.arange(0, BLOCK_SIZE)
    i = block_idx * BLOCK_SIZE
    block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
    tl.store(
        page_indices + start_idx + i + offset,
        block_ids,
        mask=i + offset < num_blocks,
    )