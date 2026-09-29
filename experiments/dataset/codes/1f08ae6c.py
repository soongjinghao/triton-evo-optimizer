import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    input_ptr,
    y_q_ptr,
    y_s_ptr,
    counts_ptr,
    H: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_RECIP: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    et = pid // (G // K + (1 if G % K else 0))
    g_base = pid % (G // K + (1 if G % K else 0)) * K
    t = et % T
    e = et // T
    n_tokens = tl.load(counts_ptr + e)
    active = t < n_tokens
    cols = tl.arange(0, BLOCK)
    g_offsets = tl.arange(0, K)
    g_valid = g_base + g_offsets < G
    h_base = (g_base + g_offsets) * GROUP_SIZE
    h_offsets = h_base[:, None] + cols[None, :]
    mask_g = g_valid[:, None] & (cols[None, :] < GROUP_SIZE) & (h_offsets < H)
    mask_g_flat = mask_g & active
    input_base = (e * T + t) * (2 * H)
    gate_flat = tl.load(
        input_ptr + input_base + h_offsets,
        mask=mask_g_flat,
        other=0.0,
    ).to(tl.float32)
    up_flat = tl.load(
        input_ptr + input_base + H + h_offsets,
        mask=mask_g_flat,
        other=0.0,
    ).to(tl.float32)
    silu_flat = gate_flat * (1.0 / (1.0 + tl.exp(-gate_flat)))
    value_flat = silu_flat * up_flat
    absmax = tl.maximum(
        tl.max(tl.abs(value_flat), axis=1),
0e-10,
    )
    if USE_RECIP:
        scale = absmax * (1.0 / 448.0)
        quant = value_flat * (448.0 / absmax)
    else:
        scale = absmax / 448.0
        quant = value_flat / scale
    quant = tl.clamp(
        quant,
        -448.0,
0,
    ).to(y_q_ptr.dtype.element_ty)
    tl.store(
        y_q_ptr + (e * T + t) * H + h_offsets,
        quant,
        mask=mask_g_flat,
    )
    tl.store(
        y_s_ptr + e * T * G + (g_base + g_offsets) * T + t,
        scale,
        mask=g_valid & active,
    )

def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_parallel_tokens: int = 16,
    group_size: int = 128,
    use_ue8m0: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert y.device.type == "npu"
    assert tokens_per_expert.device.type == "npu"
    assert y.ndim == 3
    assert group_size > 0
    if y.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float64,
    ):
        y = y.to(torch.float16)
    y_c = y if y.is_contiguous() else y.contiguous()
    counts_c = (
        tokens_per_expert
        if tokens_per_expert.is_contiguous()
        else tokens_per_expert.contiguous()
    )
    E, T, H2 = y_c.shape
    assert H2 % 2 == 0
    H = H2 // 2
    G = triton.cdiv(H, group_size)
    y_q = torch.empty(
        (E, T, H),
        dtype=torch.float16,
        device=y_c.device,
    )
    y_s = torch.empty_strided(
        (E, T, G),
        (T * G, 1, T),
        dtype=torch.float32,
        device=y_c.device,
    )
    block = triton.next_power_of_2(group_size)
    K = 4
    G_packed = G // K + (1 if G % K else 0)
    grid = (E * T * G_packed,)
    _silu_mul_fp8_quant_deep_gemm[grid](
        y_c,
        y_q,
        y_s,
        counts_c,
        H=H,
        T=T,
        G=G,
        GROUP_SIZE=group_size,
        BLOCK=block,
        USE_RECIP=False,
        K=K,
        num_warps=2,
        num_stages=1,
    )
    return y_q, y_s