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
):
    pid = tl.program_id(0)
    g = pid % G
    et = pid // G
    t = et % T
    e = et // T
    n_tokens = tl.load(counts_ptr + e)
    active = t < n_tokens
    cols = tl.arange(0, BLOCK)
    h = g * GROUP_SIZE + cols
    mask = active & (cols < GROUP_SIZE) & (h < H)
    input_base = (e * T + t) * (2 * H) + h
    gate = tl.load(
        input_ptr + input_base,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr + input_base + H,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
    value = silu * up
    absmax = tl.maximum(
        tl.max(tl.abs(value), axis=0),
        1e-10,
    )
    if USE_RECIP:
        scale = absmax * (1.0 / 448.0)
        quant = value * (448.0 / absmax)
    else:
        inv_absmax = 1.0 / absmax
        scale = absmax * (1.0 / 448.0)
        quant = value * (448.0 * inv_absmax)
    quant = tl.clamp(
        quant,
        -448.0,
0,
    ).to(y_q_ptr.dtype.element_ty)
    tl.store(
        y_q_ptr + (e * T + t) * H + h,
        quant,
        mask=mask,
    )
    tl.store(
        y_s_ptr + e * T * G + g * T + t,
        scale,
        mask=active,
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
    grid = (E * T * G,)
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
        num_warps=4,
        num_stages=2,
    )
    return y_q, y_s