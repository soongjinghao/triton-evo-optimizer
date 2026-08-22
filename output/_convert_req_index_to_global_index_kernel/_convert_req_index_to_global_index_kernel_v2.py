import torch
import triton
import triton.language as tl

@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    req = tl.load(req_id_ptr + pid)
    ti_row_off = pid * NUM_TOPK_TOKENS
    out_row_off = pid * NUM_TOPK_TOKENS
    bt_row_off = req * max_num_blocks_per_req
    token_base = tl.multiple_of(token_indices_ptr + ti_row_off, 128)
    out_base = tl.multiple_of(out_ptr + out_row_off, 128)
    block_base = tl.multiple_of(block_table_ptr + bt_row_off, 128)
    cols = tl.arange(0, BLOCK_N)
    cols = tl.max_contiguous(cols, BLOCK_N)
    tok = tl.load(token_base + cols)
    block_id = tok >> 6
    inblock_off = tok & 63
    valid_pos = tok >= 0
    valid_block = block_id < max_num_blocks_per_req
    valid_mask = valid_pos & valid_block
    base = tl.load(block_base + block_id, mask=valid_mask, other=0)
    global_pos = (base << 6) | inblock_off
    out_val = tl.where(valid_mask, global_pos, -1)
    tl.store(out_base + cols, out_val)

def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 2048,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    num_tokens = req_id.shape[0]
    req_id_c = req_id if req_id.is_contiguous() else req_id.contiguous()
    block_table_c = block_table if block_table.is_contiguous() else block_table.contiguous()
    token_indices_c = token_indices if token_indices.is_contiguous() else token_indices.contiguous()
    out = torch.empty_like(token_indices_c)
    _, max_num_blocks_per_req = block_table_c.shape
    grid = (num_tokens,)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        NUM_TOPK_TOKENS,
        num_warps=4,
        num_stages=2,
    )
    return out