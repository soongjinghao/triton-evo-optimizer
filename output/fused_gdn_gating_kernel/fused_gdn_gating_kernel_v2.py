import torch
import triton
import triton.language as tl

@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    BLK_BATCH: tl.constexpr,
):
    head_block_idx = tl.program_id(0)
    batch_block_idx = tl.program_id(1)
    head_off = head_block_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)
    batch_off = batch_block_idx * BLK_BATCH + tl.arange(0, BLK_BATCH)
    head_mask = head_off < NUM_HEADS
    batch_mask = batch_off < seq_len
    batch_mask_2d = batch_mask[:, None]
    head_mask_2d = head_mask[None, :]
    full_mask = batch_mask_2d & head_mask_2d
    batch_size = seq_len
    batch_stride = NUM_HEADS
    off_base = batch_off[:, None] * batch_stride + head_off[None, :]
    blk_A_log = tl.load(A_log + head_off, mask=head_mask, other=0.0).to(tl.float32)
    blk_bias = tl.load(dt_bias + head_off, mask=head_mask, other=0.0).to(tl.float32)
    blk_a = tl.load(a + off_base, mask=full_mask, other=0.0).to(tl.float32)
    blk_b = tl.load(b + off_base, mask=full_mask, other=0.0).to(tl.float32)
    bias_expanded = blk_bias[None, :]
    x = blk_a + bias_expanded
    beta_x = beta * x
    exp_term = tl.exp(beta_x)
    softplus_approx = (1.0 / beta) * tl.log(1.0 + exp_term)
    softplus_x = tl.where(beta_x > threshold, x, softplus_approx)
    A_exp = tl.exp(blk_A_log)
    A_exp_expanded = A_exp[None, :]
    blk_g = -A_exp_expanded * softplus_x
    tl.store(g + off_base, blk_g, mask=full_mask)
    blk_beta_output = tl.sigmoid(blk_b)
    tl.store(beta_output + off_base, blk_beta_output, mask=full_mask)

def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(a.shape) == 3:
        batch, seq_len, num_heads = a.shape
    else:
        batch, num_heads = a.shape
        seq_len = 1
    BLK_HEADS = min(triton.next_power_of_2(num_heads), 128)
    BLK_BATCH = min(triton.next_power_of_2(batch), 64)
    if BLK_HEADS * BLK_BATCH < 64:
        max_heads = min(triton.next_power_of_2(num_heads), 128)
        new_heads = min(64, max_heads)
        BLK_HEADS = max(BLK_HEADS, new_heads)
    num_head_blocks = triton.cdiv(num_heads, BLK_HEADS)
    num_batch_blocks = triton.cdiv(batch, BLK_BATCH)
    grid = (num_head_blocks, num_batch_blocks)
    if len(a.shape) == 3:
        g = torch.empty(batch, seq_len, num_heads, dtype=torch.float32, device=a.device)
        beta_output = torch.empty(batch, seq_len, num_heads, dtype=torch.float32, device=b.device)
    else:
        g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
        beta_output = torch.empty(batch, num_heads, dtype=torch.float32, device=b.device)
    total_elements_per_block = BLK_HEADS * BLK_BATCH
    if total_elements_per_block <= 512:
        num_warps = 2
    elif total_elements_per_block <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    a_2d = a.view(batch, num_heads)
    b_2d = b.view(batch, num_heads)
    fused_gdn_gating_kernel[grid](
        g.view(batch, num_heads),
        beta_output.view(batch, num_heads),
        A_log,
        a_2d,
        b_2d,
        dt_bias,
        batch,
        num_heads,
        beta,
        threshold,
        BLK_HEADS,
        BLK_BATCH,
        num_warps=num_warps,
    )
    if len(a.shape) == 3:
        g = g.view(batch, seq_len, num_heads)
        beta_output = beta_output.view(batch, seq_len, num_heads)
    return g, beta_output