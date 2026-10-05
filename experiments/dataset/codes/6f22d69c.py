import logging
from typing import Optional
import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    # Loop fully unrolled for num_experts <= 128 (compile-time constant)
    if num_experts == 1:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
    elif num_experts == 2:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
    elif num_experts == 4:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
    elif num_experts == 8:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 5 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 5 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 6 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 6 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 7 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 7 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 8 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 8 * num_experts + pid, last_cnt)
    elif num_experts == 16:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 5 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 5 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 6 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 6 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 7 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 7 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 8 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 8 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 9 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 9 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 10 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 10 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 11 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 11 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 12 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 12 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 13 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 13 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 14 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 14 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 15 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 15 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 16 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 16 * num_experts + pid, last_cnt)
    elif num_experts == 32:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 5 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 5 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 6 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 6 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 7 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 7 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 8 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 8 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 9 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 9 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 10 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 10 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 11 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 11 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 12 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 12 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 13 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 13 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 14 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 14 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 15 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 15 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 16 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 16 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 17 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 17 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 18 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 18 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 19 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 19 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 20 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 20 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 21 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 21 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 22 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 22 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 23 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 23 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 24 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 24 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 25 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 25 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 26 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 26 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 27 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 27 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 28 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 28 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 29 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 29 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 30 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 30 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 31 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 31 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 32 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 32 * num_experts + pid, last_cnt)
    elif num_experts == 64:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 5 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 5 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 6 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 6 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 7 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 7 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 8 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 8 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 9 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 9 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 10 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 10 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 11 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 11 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 12 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 12 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 13 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 13 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 14 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 14 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 15 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 15 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 16 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 16 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 17 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 17 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 18 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 18 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 19 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 19 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 20 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 20 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 21 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 21 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 22 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 22 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 23 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 23 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 24 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 24 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 25 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 25 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 26 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 26 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 27 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 27 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 28 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 28 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 29 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 29 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 30 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 30 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 31 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 31 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 32 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 32 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 33 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 33 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 34 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 34 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 35 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 35 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 36 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 36 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 37 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 37 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 38 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 38 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 39 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 39 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 40 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 40 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 41 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 41 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 42 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 42 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 43 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 43 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 44 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 44 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 45 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 45 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 46 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 46 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 47 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 47 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 48 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 48 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 49 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 49 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 50 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 50 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 51 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 51 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 52 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 52 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 53 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 53 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 54 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 54 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 55 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 55 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 56 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 56 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 57 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 57 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 58 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 58 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 59 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 59 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 60 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 60 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 61 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 61 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 62 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 62 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 63 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 63 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 64 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 64 * num_experts + pid, last_cnt)
    elif num_experts == 128:
        token_cnt = tl.load(tokens_cnts_ptr + 1 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 1 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 2 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 2 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 3 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 3 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 4 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 4 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 5 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 5 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 6 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 6 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 7 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 7 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 8 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 8 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 9 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 9 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 10 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 10 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 11 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 11 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 12 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 12 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 13 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 13 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 14 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 14 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 15 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 15 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 16 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 16 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 17 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 17 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 18 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 18 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 19 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 19 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 20 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 20 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 21 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 21 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 22 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 22 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 23 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 23 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 24 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 24 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 25 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 25 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 26 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 26 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 27 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 27 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 28 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 28 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 29 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 29 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 30 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 30 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 31 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 31 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 32 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 32 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 33 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 33 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 34 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 34 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 35 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 35 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 36 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 36 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 37 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 37 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 38 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 38 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 39 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 39 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 40 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 40 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 41 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 41 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 42 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 42 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 43 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 43 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 44 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 44 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 45 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 45 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 46 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 46 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 47 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 47 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 48 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 48 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 49 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 49 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 50 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 50 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 51 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 51 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 52 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 52 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 53 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 53 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 54 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 54 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 55 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 55 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 56 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 56 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 57 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 57 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 58 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 58 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 59 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 59 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 60 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 60 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 61 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 61 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 62 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 62 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 63 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 63 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 64 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 64 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 65 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 65 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 66 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 66 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 67 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 67 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 68 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 68 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 69 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 69 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 70 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 70 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 71 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 71 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 72 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 72 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 73 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 73 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 74 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 74 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 75 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 75 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 76 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 76 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 77 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 77 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 78 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 78 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 79 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 79 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 80 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 80 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 81 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 81 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 82 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 82 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 83 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 83 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 84 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 84 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 85 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 85 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 86 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 86 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 87 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 87 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 88 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 88 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 89 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 89 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 90 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 90 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 91 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 91 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 92 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 92 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 93 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 93 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 94 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 94 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 95 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 95 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 96 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 96 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 97 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 97 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 98 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 98 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 99 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 99 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 100 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 100 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 101 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 101 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 102 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 102 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 103 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 103 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 104 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 104 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 105 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 105 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 106 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 106 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 107 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 107 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 108 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 108 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 109 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 109 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 110 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 110 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 111 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 111 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 112 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 112 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 113 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 113 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 114 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 114 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 115 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 115 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 116 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 116 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 117 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 117 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 118 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 118 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 119 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 119 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 120 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 120 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 121 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 121 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 122 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 122 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 123 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 123 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 124 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 124 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 125 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 125 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 126 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 126 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 127 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 127 * num_experts + pid, last_cnt)
        token_cnt = tl.load(tokens_cnts_ptr + 128 * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + 128 * num_experts + pid, last_cnt)
    else:
        for i in range(1, num_experts + 1):
            token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
            last_cnt = last_cnt + token_cnt
            tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)
    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )