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
    """Wrapper with adaptive BLOCK_SIZE and num_warps optimized for Ascend NPU."""
    num_requests = cu_num_blocks.shape[0] - 1
    
    # Compute maximum number of blocks per request for adaptive sizing
    starts = cu_num_blocks[:-1]
    ends = cu_num_blocks[1:]
    max_blocks = (ends - starts).max().item()
    
    # Use fixed BLOCK_SIZE=1024 for best performance when max_blocks allows it
    # This avoids underutilization while staying within Ascend's resource limits
    if max_blocks > 512:
        BLOCK_SIZE = 1024
        num_warps = 8
    else:
        # For smaller workloads, use next power of 2 to avoid oversized blocks
        BLOCK_SIZE = max(16, triton.next_power_of_2(max_blocks))
        # Ensure multiple of 16 for Ascend alignment
        BLOCK_SIZE = ((BLOCK_SIZE + 15) // 16) * 16
        # Scale warps proportionally to hide memory latency
        num_warps = max(4, min(8, BLOCK_SIZE // 128))
    
    grid = (num_requests,)
    _copy_page_indices_kernel[grid](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )