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
    req_idx_base = tl.program_id(0) * 4
    req_offsets = tl.arange(0, 4)
    req_idx = req_idx_base + req_offsets
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx, mask=req_idx < tl.num_programs(0) * 4)
    end_idx = tl.load(cu_num_blocks + req_idx + 1, mask=req_idx < tl.num_programs(0) * 4)
    num_blocks = end_idx - start_idx
    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, 4096, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=(i + offset < num_blocks) & (req_idx < tl.num_programs(0) * 4))
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=(i + offset < num_blocks) & (req_idx < tl.num_programs(0) * 4),
        )