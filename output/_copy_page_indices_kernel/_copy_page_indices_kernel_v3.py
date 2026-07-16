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
    num_requests = cu_num_blocks.shape[0] - 1
    max_blocks = (cu_num_blocks[1:] - cu_num_blocks[:-1]).max().item()
    max_blocks = max(max_blocks, 1)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE = max(BLOCK_SIZE, 16)

    if BLOCK_SIZE >= 512:
        num_warps = 8
    elif BLOCK_SIZE >= 128:
        num_warps = 4
    else:
        num_warps = 2

    grid = (num_requests,)
    _copy_page_indices_kernel[grid](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )