import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["H"],
)
@triton.jit
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    stride: tl.constexpr = H * BT

    offset = (b * T * H + h) * BT
    A += offset
    Ai += offset

    block_row = pid_t * BT
    row_base = A + block_row * stride

    o = tl.arange(0, 16)

    D1 = tl.zeros((16, 16), tl.float32)
    D2 = tl.zeros((16, 16), tl.float32)
    D3 = tl.zeros((16, 16), tl.float32)
    D4 = tl.zeros((16, 16), tl.float32)

    for i in range(1, 16):
        r1 = -tl.load(
            row_base + i * stride + o
        ).to(tl.float32)

        r2 = -tl.load(
            row_base + (16 + i) * stride + 16 + o
        ).to(tl.float32)

        r3 = -tl.load(
            row_base + (32 + i) * stride + 32 + o
        ).to(tl.float32)

        r4 = -tl.load(
            row_base + (48 + i) * stride + 48 + o
        ).to(tl.float32)

        r1 += tl.sum(r1[:, None] * D1, 0)
        r2 += tl.sum(r2[:, None] * D2, 0)
        r3 += tl.sum(r3[:, None] * D3, 0)
        r4 += tl.sum(r4[:, None] * D4, 0)

        m = (o == i)[:, None]

        D1 = tl.where(m, r1, D1)
        D2 = tl.where(m, r2, D2)
        D3 = tl.where(m, r3, D3)
        D4 = tl.where(m, r4, D4)

    identity = o[:, None] == o[None, :]

    D1 += identity
    D2 += identity
    D3 += identity
    D4 += identity

    p21 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 16, 0),
        (16, 16), (1, 0),
    )

    p31 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 32, 0),
        (16, 16), (1, 0),
    )

    p32 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 32, 16),
        (16, 16), (1, 0),
    )

    p41 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 48, 0),
        (16, 16), (1, 0),
    )

    p42 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 48, 16),
        (16, 16), (1, 0),
    )

    p43 = tl.make_block_ptr(
        A, (T, BT), (stride, 1),
        (block_row + 48, 32),
        (16, 16), (1, 0),
    )

    A21 = tl.load(p21).to(tl.float32)
    A31 = tl.load(p31).to(tl.float32)
    A32 = tl.load(p32).to(tl.float32)
    A41 = tl.load(p41).to(tl.float32)
    A42 = tl.load(p42).to(tl.float32)
    A43 = tl.load(p43).to(tl.float32)

    P21 = tl.dot(
        D2, A21,
        input_precision=DOT_PRECISION,
    )

    P31 = tl.dot(
        D3, A31,
        input_precision=DOT_PRECISION,
    )

    P32 = tl.dot(
        D3, A32,
        input_precision=DOT_PRECISION,
    )

    P41 = tl.dot(
        D4, A41,
        input_precision=DOT_PRECISION,
    )

    P42 = tl.dot(
        D4, A42,
        input_precision=DOT_PRECISION,
    )

    P43 = tl.dot(
        D4, A43,
        input_precision=DOT_PRECISION,
    )

    X21 = -tl.dot(
        P21, D1,
        input_precision=DOT_PRECISION,
    )

    X32 = -tl.dot(
        P32, D2,
        input_precision=DOT_PRECISION,
    )

    X43 = -tl.dot(
        P43, D3,
        input_precision=DOT_PRECISION,
    )

    X31 = -(
        tl.dot(
            P31, D1,
            input_precision=DOT_PRECISION,
        )
        +
        tl.dot(
            P32, X21,
            input_precision=DOT_PRECISION,
        )
    )

    X42 = -(
        tl.dot(
            P42, D2,
            input_precision=DOT_PRECISION,
        )
        +
        tl.dot(
            P43, X32,
            input_precision=DOT_PRECISION,
        )
    )

    X41 = -(
        tl.dot(
            P41, D1,
            input_precision=DOT_PRECISION,
        )
        +
        tl.dot(
            P42, X21,
            input_precision=DOT_PRECISION,
        )
        +
        tl.dot(
            P43, X31,
            input_precision=DOT_PRECISION,
        )
    )

    q11 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row, 0),
        (16, 16), (1, 0),
    )

    q22 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 16, 16),
        (16, 16), (1, 0),
    )

    q33 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 32, 32),
        (16, 16), (1, 0),
    )

    q44 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 48, 48),
        (16, 16), (1, 0),
    )

    q21 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 16, 0),
        (16, 16), (1, 0),
    )

    q31 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 32, 0),
        (16, 16), (1, 0),
    )

    q32 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 32, 16),
        (16, 16), (1, 0),
    )

    q41 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 48, 0),
        (16, 16), (1, 0),
    )

    q42 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 48, 16),
        (16, 16), (1, 0),
    )

    q43 = tl.make_block_ptr(
        Ai, (T, BT), (stride, 1),
        (block_row + 48, 32),
        (16, 16), (1, 0),
    )

    tl.store(
        q11,
        D1.to(
            q11.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q22,
        D2.to(
            q22.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q33,
        D3.to(
            q33.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q44,
        D4.to(
            q44.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q21,
        X21.to(
            q21.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q31,
        X31.to(
            q31.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q32,
        X32.to(
            q32.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q41,
        X41.to(
            q41.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q42,
        X42.to(
            q42.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )

    tl.store(
        q43,
        X43.to(
            q43.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] == 64
    assert cu_seqlens is None

    if output_dtype is None:
        output_dtype = A.dtype

    B, T, H, BT = A.shape

    assert BT == 64
    assert T % BT == 0

    Ai = torch.zeros_like(
        A,
        dtype=output_dtype,
    )

    merge_16x16_to_64x64_inverse_kernel[
        (B * H, T // BT)
    ](
        A=A,
        Ai=Ai,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )

    return Ai