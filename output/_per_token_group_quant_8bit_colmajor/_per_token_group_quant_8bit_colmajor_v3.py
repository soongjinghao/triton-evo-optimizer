import torch
import triton
import triton.language as tl

@triton.jit
def _per_token_group_quant_8bit_colmajor_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    y_row_stride: tl.constexpr,
    y_q_row_stride: tl.constexpr,
    y_s_col_stride: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    BIT8_MIN: tl.constexpr,
    BIT8_MAX: tl.constexpr,
    REQUIRES_M_MASK: tl.constexpr,
    REQUIRES_N_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    g_start = pid_g * group_size
    col_offsets = tl.arange(0, BLOCK)

    # 统一的 mask 生成,彻底消除分支
    m_mask = m_offsets < M
    effective_n = N - g_start
    col_mask = col_offsets < effective_n
    mask2d = m_mask[:, None] & col_mask[None, :]

    # 指针计算,内层连续直接使用列偏移,无额外 stride 乘法
    in_ptrs = y_ptr + m_offsets[:, None] * y_row_stride + g_start + col_offsets[None, :]
    out_ptrs = y_q_ptr + m_offsets[:, None] * y_q_row_stride + g_start + col_offsets[None, :]
    y_s_ptrs = y_s_ptr + pid_g * y_s_col_stride + m_offsets

    # 加载及安全谓词保护
    y = tl.load(in_ptrs, mask=mask2d, other=0.0)

    # 组内量化
    _absmax = tl.max(tl.abs(y), axis=1)
    _absmax_f32 = tl.maximum(_absmax.to(tl.float32), eps)
    y_s = _absmax_f32 / BIT8_MAX
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    inv_s = 1.0 / y_s
    inv_s_native = inv_s.to(y.dtype)
    y_q = tl.clamp(y * inv_s_native[:, None], BIT8_MIN, BIT8_MAX).to(tl.int8)

    # 存回结果
    tl.store(out_ptrs, y_q, mask=mask2d)
    tl.store(y_s_ptrs, y_s, mask=m_mask)


def per_token_group_quant_8bit_colmajor(y: torch.Tensor, group_size: int, eps: float = 1e-5, scale_ue8m0: bool = False):
    assert y.is_contiguous(), "Input tensor must be contiguous"
    if y.device.type != 'npu':
        y = y.to('npu')
    y_shape = y.shape
    y_view = y.view(-1, y_shape[-1])
    M, N = y_view.shape
    num_groups_per_row = (N + group_size - 1) // group_size
    y_q = torch.empty_like(y_view, dtype=torch.int8)
    y_s = torch.empty((num_groups_per_row, M), dtype=torch.float32, device=y_view.device)
    BIT8_MIN = -128.0
    BIT8_MAX = 127.0
    BLOCK = triton.next_power_of_2(group_size)
    BLOCK = max(BLOCK, 16)
    if M >= 512:
        BLOCK_M = 32
    elif M >= 128:
        BLOCK_M = 16
    else:
        BLOCK_M = 8
    REQUIRES_M_MASK = (M % BLOCK_M != 0)
    REQUIRES_N_MASK = (N % group_size != 0) or (group_size != BLOCK)
    num_elements = BLOCK_M * BLOCK
    if num_elements >= 4096:
        num_warps = 8
    elif num_elements >= 1024:
        num_warps = 4
    else:
        num_warps = 2
    grid = (triton.cdiv(M, BLOCK_M), num_groups_per_row)
    _per_token_group_quant_8bit_colmajor_kernel[grid](
        y_view,
        y_q,
        y_s,
        group_size,
        M,
        N,
        y_view.stride(0),
        y_q.stride(0),
        y_s.stride(0),
        eps,
        BLOCK=BLOCK,
        BLOCK_M=BLOCK_M,
        SCALE_UE8M0=scale_ue8m0,
        BIT8_MIN=BIT8_MIN,
        BIT8_MAX=BIT8_MAX,
        REQUIRES_M_MASK=REQUIRES_M_MASK,
        REQUIRES_N_MASK=REQUIRES_N_MASK,
        num_warps=num_warps,
        num_stages=1,
    )
    return y_q.view(y_shape), y_s