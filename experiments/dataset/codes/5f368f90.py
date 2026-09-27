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
    num_requests = tl.num_programs(0)
    # Process 4 requests per program when num_requests % 4 == 0
    # Use a constexpr flag to enable the fused path
    # The wrapper ensures num_requests % 4 == 0 for the fused path
    # For the fused path, each program handles 4 consecutive requests
    base_req = req_idx * 4
    # Load start and end indices for all 4 requests
    start_idx_0 = tl.load(cu_num_blocks + base_req)
    end_idx_0 = tl.load(cu_num_blocks + base_req + 1)
    start_idx_1 = tl.load(cu_num_blocks + base_req + 1)
    end_idx_1 = tl.load(cu_num_blocks + base_req + 2)
    start_idx_2 = tl.load(cu_num_blocks + base_req + 2)
    end_idx_2 = tl.load(cu_num_blocks + base_req + 3)
    start_idx_3 = tl.load(cu_num_blocks + base_req + 3)
    end_idx_3 = tl.load(cu_num_blocks + base_req + 4)
    
    num_blocks_0 = end_idx_0 - start_idx_0
    num_blocks_1 = end_idx_1 - start_idx_1
    num_blocks_2 = end_idx_2 - start_idx_2
    num_blocks_3 = end_idx_3 - start_idx_3
    
    max_num_blocks = tl.maximum(tl.maximum(num_blocks_0, num_blocks_1), tl.maximum(num_blocks_2, num_blocks_3))
    
    # Vectorized address computation using tl.arange(0, BLOCK_SIZE * 4)
    offset = tl.arange(0, BLOCK_SIZE * 4)
    
    # Row pointers for each request
    row_ptr_0 = block_table + base_req * block_table_stride
    row_ptr_1 = block_table + (base_req + 1) * block_table_stride
    row_ptr_2 = block_table + (base_req + 2) * block_table_stride
    row_ptr_3 = block_table + (base_req + 3) * block_table_stride
    
    for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
        # Load block IDs for all 4 requests
        block_ids_0 = tl.load(row_ptr_0 + i + offset[:BLOCK_SIZE], mask=i + offset[:BLOCK_SIZE] < num_blocks_0)
        block_ids_1 = tl.load(row_ptr_1 + i + offset[:BLOCK_SIZE], mask=i + offset[:BLOCK_SIZE] < num_blocks_1)
        block_ids_2 = tl.load(row_ptr_2 + i + offset[:BLOCK_SIZE], mask=i + offset[:BLOCK_SIZE] < num_blocks_2)
        block_ids_3 = tl.load(row_ptr_3 + i + offset[:BLOCK_SIZE], mask=i + offset[:BLOCK_SIZE] < num_blocks_3)
        
        # Store page indices for all 4 requests
        tl.store(
            page_indices + start_idx_0 + i + offset[:BLOCK_SIZE],
            block_ids_0,
            mask=i + offset[:BLOCK_SIZE] < num_blocks_0,
        )
        tl.store(
            page_indices + start_idx_1 + i + offset[:BLOCK_SIZE],
            block_ids_1,
            mask=i + offset[:BLOCK_SIZE] < num_blocks_1,
        )
        tl.store(
            page_indices + start_idx_2 + i + offset[:BLOCK_SIZE],
            block_ids_2,
            mask=i + offset[:BLOCK_SIZE] < num_blocks_2,
        )
        tl.store(
            page_indices + start_idx_3 + i + offset[:BLOCK_SIZE],
            block_ids_3,
            mask=i + offset[:BLOCK_SIZE] < num_blocks_3,
        )