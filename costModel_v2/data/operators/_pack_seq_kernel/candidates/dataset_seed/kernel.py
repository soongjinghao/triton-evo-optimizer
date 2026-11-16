import torch
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    lengths_ptr,
    D: tl.constexpr,
    SEQ_ELEMENTS: tl.constexpr, # 💡 常量折叠，消灭内核乘法
    PAD_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REQUIRES_STORE_MASK: tl.constexpr,
):
    pid_flat = tl.program_id(0)
    pid_b = tl.program_id(1)

    # 1. 并发发射两路标量读请求
    start_idx = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    # 2. 独立执行地址向量化计算，利用底层流水线隐藏上方访存延迟
    offs = pid_flat * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_base = start_idx * D
    dst_base = pid_b * SEQ_ELEMENTS
    
    src_ptrs = x_ptr + src_base + offs
    dst_ptrs = out_ptr + dst_base + offs

    # 3. 计算实际有效读取边界
    valid_elements = seq_len * D
    load_mask = offs < valid_elements

    # 4. 静态剥离冗余写掩码，极速 DMA 搬运
    if not REQUIRES_STORE_MASK:
        vals = tl.load(src_ptrs, mask=load_mask, other=PAD_VALUE)
        tl.store(dst_ptrs, vals)
    else:
        store_mask = offs < SEQ_ELEMENTS
        vals = tl.load(src_ptrs, mask=load_mask, other=PAD_VALUE)
        tl.store(dst_ptrs, vals, mask=store_mask)

def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64, 
    block_d: int = 64, 
    use_precomputed_starts: bool = True,
) -> torch.Tensor:
    original_shape = x.shape
    
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.view(N, -1)  # 💡 强制使用 view 避免 reshape 潜在的内存拷贝开销
        D = x_reshaped.shape[1]
    else:
        N, D = original_shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    if use_precomputed_starts:
        # =================================================================
        # 💡 绝杀：消灭 Python 层面的端侧调度黑洞！
        # 仅用一行底层连续向量化计算完成，彻底扫清 Host 端的 4 次下发延迟！
        # =================================================================
        starts = lengths.cumsum(0, dtype=torch.int32) - lengths
    else:
        starts = torch.zeros_like(lengths, dtype=torch.int32)

    # 💡 常量提权：预先算好总元素，传入 constexpr
    SEQ_ELEMENTS = Lmax * D
    
    BLOCK_SIZE = triton.next_power_of_2(SEQ_ELEMENTS)
    BLOCK_SIZE = min(max(BLOCK_SIZE, 256), 4096)

    grid = (triton.cdiv(SEQ_ELEMENTS, BLOCK_SIZE), B)
    REQUIRES_STORE_MASK = (SEQ_ELEMENTS % BLOCK_SIZE != 0)

    if BLOCK_SIZE >= 4096:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    else:
        num_warps = 2

    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        starts,
        lengths,
        D=D,
        SEQ_ELEMENTS=SEQ_ELEMENTS,
        PAD_VALUE=float(pad_value),
        BLOCK_SIZE=BLOCK_SIZE,
        REQUIRES_STORE_MASK=REQUIRES_STORE_MASK,
        num_warps=num_warps,
        num_stages=1, 
    )

    if len(original_shape) > 2:
        out = out.view((B, Lmax) + original_shape[1:])

    return out