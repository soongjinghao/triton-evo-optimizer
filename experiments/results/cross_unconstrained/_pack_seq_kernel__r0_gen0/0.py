import torch
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    in_start: tl.int32,
    seq_len: tl.int32,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t_i32 = pid_t.to(tl.int32)
    pid_d_i32 = pid_d.to(tl.int32)
    off_t = pid_t_i32 * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int32)
    off_d = pid_d_i32 * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int32)
    t_mask = off_t < Lmax
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
    out_row_ptr = out_ptr + (pid_t_i32 * BLOCK_T + off_t)[:, None] * D + off_d[None, :]
    d_mask = off_d[None, :] < D
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

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
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x
    B = lengths.numel()
    Lmax = int(lengths.max().item())
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)
    if use_precomputed_starts:
        starts = torch.zeros_like(lengths)
        if B > 1:
            starts[1:] = torch.cumsum(lengths[:-1], dim=0)
        starts = starts.int()
    else:
        starts = torch.zeros_like(lengths, dtype=torch.int32)
    lengths_int = lengths.int()
    for b in range(B):
        cur_start = starts[b].item()
        cur_len = lengths_int[b].item()
        grid = (triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
        _pack_seq_kernel[grid](
            x_reshaped,
            out,
            N,
            D,
            Lmax,
            PAD_VALUE=float(pad_value),
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            in_start=cur_start,
            seq_len=cur_len,
            num_warps=4,
            num_stages=2,
        )
    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)
    return out