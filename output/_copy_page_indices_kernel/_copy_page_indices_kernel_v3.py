import triton
import triton.language as tl
from triton import Config

@triton.autotune(
    configs=[
        Config(kwargs={}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [4, 8]
    ],
    key=[],
)
@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)
    src_ptr = block_table + pid * block_table_stride
    src_ptr = tl.multiple_of(src_ptr, 16)
    load_mask = offs < block_table_stride
    block_ids = tl.load(src_ptr + offs, mask=load_mask)
    cu_ptr = cu_num_blocks + pid
    start_idx = tl.load(cu_ptr)
    end_idx = tl.load(cu_ptr + 1)
    num_blocks = end_idx - start_idx
    dst_ptr = page_indices + start_idx
    dst_ptr = tl.multiple_of(dst_ptr, 16)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)
    store_mask = offs < num_blocks
    tl.store(dst_ptr + offs, block_ids, mask=store_mask)

def copy_page_indices(page_indices, block_table, block_table_stride, cu_num_blocks, BLOCK_SIZE=128):
    grid = (cu_num_blocks.numel() - 1,)
    _copy_page_indices_kernel[grid](page_indices, block_table, block_table_stride, cu_num_blocks, BLOCK_SIZE)
    return page_indices