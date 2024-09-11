import triton
import torch

import triton.language as tl

@triton.jit
def one_to_n_loads_full(In, GatherIndx, Out, stride_x_m, stride_x_k, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    offs_x_m = tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    offs_x_m = tl.load(GatherIndx + offs_x_m) // 7

    offs_x_k = tl.arange(0, BLOCK_K)

    InPtrs = In + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    mask_k = tl.arange(0, BLOCK_K) < k

    # Do we ever mask out the rows? Or only the columns?
    x = tl.load(InPtrs, mask=mask_k[None, :], other=0.0)

    OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    tl.store(OutPtrs, x)


@triton.jit
def one_to_n_loads_simplified(In, GatherIndx, Out, stride_x_m, stride_x_k, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    # 1. Assume no masks for simplicity
    # 2. Simplified the first load (no modulo)
    # 3. Simplified no div
    offs_x_m = tl.arange(0, BLOCK_M)
    offs_x_m = tl.load(GatherIndx + offs_x_m)

    offs_x_k = tl.arange(0, BLOCK_K)

    InPtrs = In + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
    OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    for k in range(0, 2):
        x = tl.load(InPtrs)
        tl.store(OutPtrs, x)
        InPtrs += BLOCK_K * stride_x_k
        OutPtrs += BLOCK_K * stride_x_k


@triton.jit
def one_to_n_loads_simplified_flipped(In, GatherIndx, Out, stride_x_m, stride_x_k, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    # 1. Assume no masks for simplicity
    # 2. Simplified the first load (no modulo)
    # 3. Simplified no div
    offs_x_m = tl.arange(0, BLOCK_M)
    offs_x_m = tl.load(GatherIndx + offs_x_m)

    offs_x_k = tl.arange(0, BLOCK_K)

    InPtrs = In + offs_x_k[None, :] * stride_x_k + offs_x_m[:, None] * stride_x_m
    OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    for k in range(0, 2):
        x = tl.load(InPtrs)
        tl.store(OutPtrs, x)
        InPtrs += BLOCK_K * stride_x_k
        OutPtrs += BLOCK_K * stride_x_k

@triton.jit
def one_to_n_loads_no_loops(In, GatherIndx, Out, stride_x_m, stride_x_k, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    # 1. Assume no masks for simplicity
    # 2. Simplified the first load (no modulo)
    # 3. Simplified no div
    offs_x_m = tl.arange(0, BLOCK_M)
    offs_x_m = tl.load(GatherIndx + offs_x_m)

    offs_x_k = tl.arange(0, BLOCK_K)

    InPtrs = In + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    x = tl.load(InPtrs)
    tl.store(OutPtrs, x)


@triton.jit
def test(In, GatherIndx, Out, stride_x_m, stride_x_k, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    # 1. Assume no masks for simplicity
    # 2. Simplified the first load (no modulo)
    # 3. Simplified no div
    offs_x_m = tl.arange(0, BLOCK_M)
    offs_x_k = tl.arange(0, BLOCK_K)

    InPtrs = In + offs_x_m[:, None] + offs_x_k[None, :] * stride_x_k

    OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k

    x = tl.load(InPtrs)
    tl.store(OutPtrs, x)



def main():
    src = triton.compiler.ASTSource(
        fn=one_to_n_loads_simplified_flipped,
        signature="*fp32,*i32,*fp32,i32,i32",
        constants={
            "BLOCK_M": 2,
            "BLOCK_K": 4
        }
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

main()
