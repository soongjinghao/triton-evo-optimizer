import torch
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    lengths_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    in_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)
    
    t_mask = off_t < Lmax
    d_mask = off_d < D
    valid_row = (off_t < seq_len) & t_mask
    
    # Hoist: precompute D and Lmax*D as loop invariants
    D_const = D.to(tl.int32)
    LmaxD = (Lmax * D).to(tl.int32)
    
    # Base offsets for input and output rows
    in_base = (in_start * D_const).to(tl.int32)
    out_base = (pid_b * LmaxD).to(tl.int32)
    
    # Compute row offsets using hoisted constants
    in_row_offsets = in_base + off_t * D_const
    out_row_offsets = out_base + off_t * D_const
    
    # Expand to 2D with column offsets
    in_row_ptr = x_ptr + in_row_offsets[:, None] + off_d[None, :]
    out_row_ptr = out_ptr + out_row_offsets[:, None] + off_d[None, :]
    
    # Add alignment hints for D and Lmax*D (both aligned to 128B when D is multiple of 32 float32 elements)
    # D_const and LmaxD are compile-time constants, so tl.multiple_of is valid
    in_row_ptr = tl.multiple_of(in_row_ptr, (D_const,))
    out_row_ptr = tl.multiple_of(out_row_ptr, (LmaxD,))
    
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
    x_vals = tl.load(in_row_ptr, mask=valid_row[:, None] & d_mask)
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
    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        starts.int(),
        lengths.int(),
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