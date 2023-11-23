import torch

import triton
import triton.language as tl

@triton.jit
def wrap_side_by_side_loop(
    a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE_K: tl.constexpr
):
    offs_am = tl.arange(0, 4)
    offs_an = (tl.arange(0, 4)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)

    offs_k = tl.arange(0, 4)

    offs_cm = tl.arange(0, 4)
    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    for k in range(0, 3):
        a = tl.load(a_ptrs)
        tl.store(c_ptrs, a)
        a_ptrs += BLOCK_SIZE_K * stride_am
        c_ptrs += BLOCK_SIZE_K * stride_an



@triton.jit
def wrap_side_by_side_loop_unroll(
    a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE_K: tl.constexpr
):
    offs_am = tl.arange(0, 4)
    offs_an = (6 + tl.arange(0, 4)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)

    offs_k = tl.arange(0, 4)

    offs_cm = tl.arange(0, 4)
    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    a = tl.load(a_ptrs)
    tl.store(c_ptrs, a)
    a_ptrs += BLOCK_SIZE_K * stride_am
    c_ptrs += BLOCK_SIZE_K * stride_an

    a = tl.load(a_ptrs)
    tl.store(c_ptrs, a)
    a_ptrs += BLOCK_SIZE_K * stride_am
    c_ptrs += BLOCK_SIZE_K * stride_an

    a = tl.load(a_ptrs)
    tl.store(c_ptrs, a)
    a_ptrs += BLOCK_SIZE_K * stride_am
    c_ptrs += BLOCK_SIZE_K * stride_an


@triton.jit
def mod_1d(
    a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE_K: tl.constexpr
):
    row = 7
    offs_an = (6 + tl.arange(0, 4)) % N
    a_ptrs = a_ptr + (row * stride_am) + offs_an[None, :] * stride_an

    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cn * offs_cn[None, :]

    a = tl.load(a_ptrs)
    tl.store(c_ptrs, a)


@triton.jit
def mod_2d(
    a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE_K: tl.constexpr
):
    k = 1111
    h = 1
    k = 1222222
    www = 13
    offs_am = 2 + tl.arange(0, 4)
    offs_an = (6 + tl.arange(0, 4)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)

    offs_cm = tl.arange(0, 4)
    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    a = tl.load(a_ptrs)
    tl.store(c_ptrs, a)


def test():
    M = 12
    N = 8
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    A = torch.arange(0, M * N, device="cpu", dtype=torch.float32).reshape((M, N))
    out = torch.full((BLOCK_SIZE_M, 12), 88888, device="cpu", dtype=torch.float32)
    print(out)
    grid = lambda meta: (1,)

    wrap_side_by_side_loop_unroll[grid](
        A,
        out,
        M,
        N,
        A.stride(0),
        A.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_K=4
    )

    expected_out = torch.tensor(
        [
            [22.0, 23.0, 16.0, 17.0, 54.0, 55.0, 48.0, 49.0],
            [30.0, 31.0, 24.0, 25.0, 62.0, 63.0, 56.0, 57.0],
            [-99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0],
            [-99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0],
        ]
    )

    print(A)
    print(out.int())
    # assert torch.equal(expected_out.int(), out.int())


def compile():
    ret = triton.compile(wrap_side_by_side_masked_loop, signature="*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32")
    print(ret.asm["ttir"])


test()
# compile()
