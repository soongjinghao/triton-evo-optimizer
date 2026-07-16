import torch
from vllm.triton_utils import tl, triton
from typing import Optional

@triton.jit
def _silu_mul_fp8_quant_hybrid(
    input_ptr,
    y_q_ptr,
    y_s_ptr,
    counts_ptr,
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    stride_counts_e,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    # Dynamic mask: handles partial last group correctly
    global_col = g * GROUP_SIZE + cols
    mask = global_col < H

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0
        )

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Simple max reduction (supported on NPU)
        max_abs = tl.max(tl.abs(y))
        y_s = tl.maximum(max_abs, eps) / fp8_max
        if use_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_parallel_tokens=16,
    group_size: int = 256,  # mutated: increased from 128 to 256 for better occupancy
    use_ue8m0: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("npu")
    assert y.device.type == "npu", "All tensors must be on NPU"
    assert tokens_per_expert.device.type == "npu"

    # Convert unsupported dtypes to float16 for NPU compatibility
    if y.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float64]:
        y = y.to(torch.float16)

    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2

    # Output buffer: float16 as proxy for FP8 on NPU
    y_q = torch.empty((E, T, H), dtype=torch.float16, device=device)

    G = (H + group_size - 1) // group_size
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided(
        (E, T, G),
        (stride_ys_e, stride_ys_t, stride_ys_g),
        dtype=torch.float32,
        device=device,
    )

    use_ue8m0_val = False  # UE8M0 not available on NPU

    stride_cnt_e = tokens_per_expert.stride()[0]
    grid = (E * G,)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    fp8_max = 448.0   # E4M3 max
    fp8_min = -448.0  # E4M3 min
    eps: float = 1e-10

    _silu_mul_fp8_quant_hybrid[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_ys_e,
        stride_ys_t,
        stride_ys_g,
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0_val,
        BLOCK=group_size,          # now 256, matches 8 warps * 32 threads
        NUM_STAGES=4,              # moderate pipeline depth to balance latency and registers
        num_warps=8,               # increased warp count to hide latency
    )

    return y_q, y_s