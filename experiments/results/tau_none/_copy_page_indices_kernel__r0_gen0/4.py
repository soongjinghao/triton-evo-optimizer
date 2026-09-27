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
    pid = tl.program_id(0)
    req_idx = pid * 4
    for req in range(4):
        idx = req_idx + req
        row_ptr = block_table + idx * block_table_stride
        start_idx = tl.load(cu_num_blocks + idx)
        end_idx = tl.load(cu_num_blocks + idx + 1)
        num_blocks = end_idx - start_idx
        offset = tl.arange(0, BLOCK_SIZE)
        for i in tl.range(0, num_blocks, BLOCK_SIZE):
            block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
            tl.store(
                page_indices + start_idx + i + offset,
                block_ids,
                mask=i + offset < num_blocks,
            )