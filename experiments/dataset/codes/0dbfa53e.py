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
    req_idx = pid // BLOCK_SIZE
    local_offset = pid % BLOCK_SIZE
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx
    if local_offset < num_blocks:
        block_id = tl.load(row_ptr + local_offset)
        tl.store(page_indices + start_idx + local_offset, block_id)