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
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )

def copy_page_indices(page_indices, block_table, cu_num_blocks):
    """Wrapper that adaptively sets BLOCK_SIZE and num_warps for Ascend NPU."""
    num_requests = cu_num_blocks.shape[0] - 1
    # Fixed BLOCK_SIZE=1024 (multiple of 16) and num_warps=8 per Winner #1 strategy.
    # This over-allocates resources but is the best among considered options.
    BLOCK_SIZE = 1024
    num_warps = 8
    grid = (num_requests,)
    _copy_page_indices_kernel[grid](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )