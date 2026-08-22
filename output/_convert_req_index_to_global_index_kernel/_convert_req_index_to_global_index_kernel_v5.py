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
    # 使用 NUM_TOPK_TOKENS 作为实际分块宽度,保证一次处理完所有 topk token
    cols = tl.arange(0, NUM_TOPK_TOKENS)
    # 计算本请求对应的 token_indices / out 的基地址
    ti_base = token_indices_ptr + pid * NUM_TOPK_TOKENS
    out_base = out_ptr + pid * NUM_TOPK_TOKENS
    # 块表基地址
    bt_base = block_table_ptr + req * max_num_blocks_per_req

    # 向量化加载 topk 索引
    token = tl.load(ti_base + cols, mask=cols < NUM_TOPK_TOKENS, other=-1)
    block_id = token >> 6
    inblock_off = token & 63
    # 合法性掩码:非负 token 且 block_id 未越界
    valid = (token >= 0) & (block_id < max_num_blocks_per_req)
    # 从块表中加载 base 地址
    base = tl.load(bt_base + block_id, mask=valid, other=0)
    # 合成全局位置
    global_pos = (base << 6) | inblock_off
    out_val = tl.where(valid, global_pos, -1)
    # 向量化写回
    tl.store(out_base + cols, out_val, mask=cols < NUM_TOPK_TOKENS)


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
    # 1D 网格,每个请求由一个 program 处理,完全消除串行循环
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