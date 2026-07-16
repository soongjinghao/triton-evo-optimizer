import torch
from vllm.triton_utils import tl, triton
from typing import Optional

@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
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

    # cols range over the padded tile; valid indices are cols < GROUP_SIZE
    cols = tl.arange(0, BLOCK).to(tl.int64)
    valid_mask = cols < GROUP_SIZE

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    # Unroll token loop by 4x for higher throughput
    STEP: tl.constexpr = 4
    for t in tl.range(0, n_tokens, step=STEP, num_stages=NUM_STAGES):
        t0 = t
        t1 = t + 1
        t2 = t + 2
        t3 = t + 3
        t1_valid = t1 < n_tokens
        t2_valid = t2 < n_tokens
        t3_valid = t3 < n_tokens

        # Load gate and up for token t0 (always valid)
        gate0 = tl.load(
            input_ptr + base_gate_offset + t0 * stride_i_t,
            mask=valid_mask, other=0.0
        ).to(tl.float32)
        up0 = tl.load(
            input_ptr + base_up_offset + t0 * stride_i_t,
            mask=valid_mask, other=0.0
        )

        # Load gate and up for token t1
        gate1 = tl.load(
            input_ptr + base_gate_offset + t1 * stride_i_t,
            mask=t1_valid & valid_mask, other=0.0
        ).to(tl.float32)
        up1 = tl.load(
            input_ptr + base_up_offset + t1 * stride_i_t,
            mask=t1_valid & valid_mask, other=0.0
        )

        # Load gate and up for token t2
        gate2 = tl.load(
            input_ptr + base_gate_offset + t2 * stride_i_t,
            mask=t2_valid & valid_mask, other=0.0
        ).to(tl.float32)
        up2 = tl.load(
            input_ptr + base_up_offset + t2 * stride_i_t,
            mask=t2_valid & valid_mask, other=0.0
        )

        # Load gate and up for token t3
        gate3 = tl.load(
            input_ptr + base_gate_offset + t3 * stride_i_t,
            mask=t3_valid & valid_mask, other=0.0
        ).to(tl.float32)
        up3 = tl.load(
            input_ptr + base_up_offset + t3 * stride_i_t,
            mask=t3_valid & valid_mask, other=0.0
        )

        # SiLU and multiply for all tokens
        gate0 = gate0 * (1.0 / (1.0 + tl.exp(-gate0)))
        y0 = gate0 * up0

        gate1 = gate1 * (1.0 / (1.0 + tl.exp(-gate1)))
        y1 = gate1 * up1

        gate2 = gate2 * (1.0 / (1.0 + tl.exp(-gate2)))
        y2 = gate2 * up2

        gate3 = gate3 * (1.0 / (1.0 + tl.exp(-gate3)))
        y3 = gate3 * up3

        # Quantization for token t0
        y_s0 = tl.maximum(tl.max(tl.abs(y0)), eps) / fp8_max
        if use_ue8m0:
            y_s0 = tl.exp2(tl.ceil(tl.log2(y_s0)))
        y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        # Quantization for token t1
        y_s1 = tl.maximum(tl.max(tl.abs(y1)), eps) / fp8_max
        if use_ue8m0:
            y_s1 = tl.exp2(tl.ceil(tl.log2(y_s1)))
        y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        # Quantization for token t2
        y_s2 = tl.maximum(tl.max(tl.abs(y2)), eps) / fp8_max
        if use_ue8m0:
            y_s2 = tl.exp2(tl.ceil(tl.log2(y_s2)))
        y_q2 = tl.clamp(y2 / y_s2, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        # Quantization for token t3
        y_s3 = tl.maximum(tl.max(tl.abs(y3)), eps) / fp8_max
        if use_ue8m0:
            y_s3 = tl.exp2(tl.ceil(tl.log2(y_s3)))
        y_q3 = tl.clamp(y3 / y_s3, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        # Store results for token t0
        tl.store(y_q_ptr + base_yq_offset + t0 * stride_yq_t, y_q0, mask=valid_mask)
        tl.store(y_s_ptr + base_ys_offset + t0 * stride_ys_t, y_s0)

        # Store for token t1 (masked)
        tl.store(y_q_ptr + base_yq_offset + t1 * stride_yq_t, y_q1,
                 mask=t1_valid & valid_mask)
        tl.store(y_s_ptr + base_ys_offset + t1 * stride_ys_t, y_s1,
                 mask=t1_valid)

        # Store for token t2 (masked)
        tl.store(y_q_ptr + base_yq_offset + t2 * stride_yq_t, y_q2,
                 mask=t2_valid & valid_mask)
        tl.store(y_s_ptr + base_ys_offset + t2 * stride_ys_t, y_s2,
                 mask=t2_valid)

        # Store for token t3 (masked)
        tl.store(y_q_ptr + base_yq_offset + t3 * stride_yq_t, y_q3,
                 mask=t3_valid & valid_mask)
        tl.store(y_s_ptr + base_ys_offset + t3 * stride_ys_t, y_s3,
                 mask=t3_valid)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_parallel_tokens=16,
    group_size: int = 128,
    use_ue8m0: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("npu")
    assert y.device.type == "npu", "All tensors must be on NPU"
    assert tokens_per_expert.device.type == "npu"
    
    # Convert unsupported dtypes
    if y.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float64]:
        y = y.to(torch.float16)
    
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    
    # Use float16 instead of FP8 for NPU
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

    use_ue8m0_val = False

    stride_cnt_e = tokens_per_expert.stride()[0]
    grid = (E * G,)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
    
    fp8_max = 448.0
    fp8_min = -448.0
    eps: float = 1e-10

    # Enforce BLOCK to be a multiple of 16 for Ascend Cube alignment
    BLOCK_ALIGNED = ((group_size + 15) // 16) * 16
    
    _silu_mul_fp8_quant_deep_gemm[grid](
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
        BLOCK=BLOCK_ALIGNED,
        NUM_STAGES=4,
        num_warps=8,
    )
    
    return y_q, y_s