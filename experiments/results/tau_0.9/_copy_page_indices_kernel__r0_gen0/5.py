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
    num_blocks = tl.load(cu_num_blocks + req_idx + 1) - tl.load(cu_num_blocks + req_idx)
    start_idx = tl.load(cu_num_blocks + req_idx)
    offset = tl.arange(0, BLOCK_SIZE)
    row_ptr = block_table + req_idx * num_blocks
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + offset, mask=offset < num_blocks - i, other=0)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
        row_ptr += BLOCK_SIZE