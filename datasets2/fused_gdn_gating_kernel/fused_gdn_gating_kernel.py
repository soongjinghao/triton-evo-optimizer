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
    batch_size,
    seq_len,
    NUM_HEADS: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    BATCHES_PER_BLOCK: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
):

    block_idx = tl.program_id(0)

    batch_start = block_idx * BATCHES_PER_BLOCK

    if batch_start >= batch_size:
        return

    batch_end = tl.minimum(batch_start + BATCHES_PER_BLOCK, batch_size)
    batches_in_block = batch_end - batch_start

    num_head_chunks = tl.cdiv(NUM_HEADS, BLK_HEADS)

    for chunk_idx in range(num_head_chunks):
        h_start = chunk_idx * BLK_HEADS
        h_end = tl.minimum(h_start + BLK_HEADS, NUM_HEADS)

        h_idx = tl.arange(0, BLK_HEADS)
        head_global_idx = h_start + h_idx
        head_mask = head_global_idx < NUM_HEADS

        blk_A_log = tl.load(A_log + head_global_idx, mask=head_mask)
        blk_bias = tl.load(dt_bias + head_global_idx, mask=head_mask)

        for b_off in range(batches_in_block):
            batch_idx = batch_start + b_off

            base_offset = batch_idx * NUM_HEADS + h_start
            head_offsets = base_offset + h_idx

            combined_mask = head_mask

            blk_a = tl.load(a + head_offsets, mask=combined_mask)
            blk_b = tl.load(b + head_offsets, mask=combined_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
            beta_x = beta * x

            softplus_x = tl.where(
                beta_x <= threshold,
                (1 / beta) * tl.log(1 + tl.exp(beta_x)),
                x
            )
            
            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x

            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))

            tl.store(g + head_offsets, blk_g.to(g.dtype.element_ty), mask=combined_mask)
            tl.store(
                beta_output + head_offsets,
                blk_beta_output.to(beta_output.dtype.element_ty),
                mask=combined_mask
            )

def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    max_processes: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net optimized for Ascend NPU.
    
    Args:
        A_log: [NUM_HEADS]
        a: [batch_size, NUM_HEADS]
        b: [batch_size, NUM_HEADS]
        dt_bias: [NUM_HEADS]
        max_processes: Maximum number of kernel calls (Ascend NPU limit)
    
    Returns:
        g: [1, batch_size, NUM_HEADS]
        beta_output: [1, batch_size, NUM_HEADS]
    """
    batch_size, num_heads = a.shape

    g = torch.empty(1, batch_size, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch_size, num_heads, dtype=b.dtype, device=b.device)

    BLK_HEADS = 16
    BATCHES_PER_BLOCK = triton.cdiv(batch_size, max_processes)
    num_blocks = triton.cdiv(batch_size, BATCHES_PER_BLOCK)

    num_blocks = min(num_blocks, max_processes)
    
    print(f"NPU Kernel Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Batches per block: {BATCHES_PER_BLOCK}")
    print(f"  Num blocks (kernel calls): {num_blocks}")
    print(f"  Max allowed processes: {max_processes}")

    fused_gdn_gating_kernel[(num_blocks,)](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        batch_size,
        1,
        num_heads,
        BLK_HEADS,
        BATCHES_PER_BLOCK,
        beta,
        threshold,
        num_warps=4,
    )
    
    return g, beta_output

