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
    bt_stride0,
    ti_stride0,
    out_stride0,
    NUM_TOPK_TOKENS: tl.constexpr,
    ):
    pid = tl.program_id(0)
    # 1. 物理行基址计算
    req = tl.load(req_id_ptr + pid)
    ti_row_off = pid * ti_stride0
    out_row_off = pid * out_stride0
    bt_row_off = req * bt_stride0
    token_base = tl.multiple_of(token_indices_ptr + ti_row_off, 16)
    out_base = tl.multiple_of(out_ptr + out_row_off, 16)
    block_base = tl.multiple_of(block_table_ptr + bt_row_off, 16)
    # 2. 一次性打平加载 2048 个 Token Indices
    cols = tl.arange(0, BLOCK_N)
    cols = tl.max_contiguous(cols, BLOCK_N)
    tok = tl.load(token_base + cols)
    # 3. 极速位提取
    block_id = tok >> 6
    inblock_off = tok & 63
    # 4. 单侧掩码剪枝（省去冗余条件比较）
    valid_mask = (tok >= 0) & (block_id < max_num_blocks_per_req)
    # 5. 原生 Mask 守护的离散内存 Gather
    base = tl.load(block_base + block_id, mask=valid_mask, other=0)
    # 6. 位运算合成物理索引并写回
    global_pos = (base << 6) | inblock_off
    out_val = tl.where(valid_mask, global_pos, -1)
    tl.store(out_base + cols, out_val)
def triton_convert_req_index_to_global_index( 
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 2048,  # 最佳性能配置
    ):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    num_tokens = req_id.shape[0]
    # Host 侧连续性判断，避免重复 allocate
    req_id_c = req_id if req_id.is_contiguous() else req_id.contiguous()
    block_table_c = block_table if block_table.is_contiguous() else block_table.contiguous()
    token_indices_c = token_indices if token_indices.is_contiguous() else token_indices.contiguous()
    out = torch.empty_like(token_indices_c)
    bt_stride0 = block_table_c.stride(0)
    ti_stride0 = token_indices_c.stride(0)
    out_stride0 = out.stride(0)
    _, max_num_blocks_per_req = block_table.shape
    grid = (num_tokens,)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        bt_stride0,
        ti_stride0,
        out_stride0,
        NUM_TOPK_TOKENS,
        # 💡 仅调整 launch 参数，探索编译器流水线最优配置
        # 针对处理 2048 个 int32 元素，4 个 warps 和 2 级流水线通常能取得较好的平衡
        num_warps=4,
        num_stages=2, 
    )
    return out