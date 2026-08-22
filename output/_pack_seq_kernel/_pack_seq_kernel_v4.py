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
    SEQ_ELEMENTS: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REQUIRES_STORE_MASK: tl.constexpr,
):
    pid_flat = tl.program_id(0)
    pid_b = tl.program_id(1)

    start_idx = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    offs = pid_flat * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_base = start_idx * D
    dst_base = pid_b * SEQ_ELEMENTS

    # 宣告基地址 16 字节对齐,启用向量化突发传输
    src_ptrs = tl.multiple_of(x_ptr + src_base, 16) + offs
    dst_ptrs = tl.multiple_of(out_ptr + dst_base, 16) + offs

    valid_elements = seq_len * D
    load_mask = offs < valid_elements

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
        x_reshaped = x.view(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = original_shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    # 向量化对齐前置条件检查
    assert x_reshaped.data_ptr() % 16 == 0, "x_reshaped pointer must be 16-byte aligned"
    assert out.data_ptr() % 16 == 0, "out pointer must be 16-byte aligned"
    assert (D * x.element_size()) % 16 == 0, "D * element_size must be multiple of 16 for alignment"

    if use_precomputed_starts:
        starts = lengths.cumsum(0, dtype=torch.int32) - lengths
    else:
        starts = torch.zeros_like(lengths, dtype=torch.int32)

    SEQ_ELEMENTS = Lmax * D

    # 计算 BLOCK_SIZE,确保为 128 的倍数以匹配向量宽度
    BLOCK_SIZE = triton.next_power_of_2(SEQ_ELEMENTS)
    BLOCK_SIZE = min(max(BLOCK_SIZE, 256), 4096)
    BLOCK_SIZE = (BLOCK_SIZE + 127) // 128 * 128

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