import torch
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    lengths_ptr,
    cum_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    in_start = tl.load(cum_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = off_d < D
    for off_t in range(0, Lmax, BLOCK_T):
        t_offs = off_t + tl.arange(0, BLOCK_T)
        t_mask = t_offs < Lmax
        valid_row = (t_offs < seq_len) & t_mask
        in_row = in_start + t_offs
        x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
        out_row_ptr = out_ptr + (pid_b * Lmax + t_offs)[:, None] * D + off_d[None, :]
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
        tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
        x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
        tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

@triton.jit
def _cum_lengths_kernel(
    lengths_ptr,
    cum_ptr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(cum_ptr + 0, 0)
    else:
        cum_val = 0
        for i in range(pid):
            cum_val += tl.load(lengths_ptr + i)
        tl.store(cum_ptr + pid, cum_val)

def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x
    B = lengths.numel()
    Lmax = int(lengths.max().item())
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)
    cum_lengths = torch.empty(B, device=x.device, dtype=torch.int32)
    grid_cum = (B,)
    _cum_lengths_kernel[grid_cum](lengths.int(), cum_lengths, B)
    grid = (B, triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        cum_lengths,
        N,
        D,
        Lmax,
        PAD_VALUE=float(pad_value),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)
    return out