import torch

import triton
import triton.language as tl

@triton.jit
def wrap_side_by_side_loop(
    a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn, BLOCK_SIZE_K: tl.constexpr
):
    offs_am = tl.arange(0, 4)
    offs_an = (6 + tl.arange(0, 4)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
    k = 12322111111111
    offs_k = tl.arange(0, 4)

    offs_cm = tl.arange(0, 4)
    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    for k in range(0, 3):
        a = tl.load(a_ptrs)
        tl.store(c_ptrs, a)
        a_ptrs += BLOCK_SIZE_K * stride_am
        c_ptrs += BLOCK_SIZE_K * stride_am



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
    BLOCK_SIZE_M = 5
    BLOCK_SIZE_N = 4
    A = torch.arange(0, M * N, device="cpu", dtype=torch.float32).reshape((M, N))
    out = torch.full((M, N), 88888, device="cpu", dtype=torch.float32)
    print(out)
    grid = lambda meta: (1,)

    wrap_side_by_side_loop[grid](
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

    expected_out = torch.tensor([[    6,     7,     0,     1, 88888, 88888, 88888, 88888],
        [   14,    15,     8,     9, 88888, 88888, 88888, 88888],
        [   22,    23,    16,    17, 88888, 88888, 88888, 88888],
        [   30,    31,    24,    25, 88888, 88888, 88888, 88888],
        [   38,    39,    32,    33, 88888, 88888, 88888, 88888],
        [   46,    47,    40,    41, 88888, 88888, 88888, 88888],
        [   54,    55,    48,    49, 88888, 88888, 88888, 88888],
        [   62,    63,    56,    57, 88888, 88888, 88888, 88888],
        [   70,    71,    64,    65, 88888, 88888, 88888, 88888],
        [   78,    79,    72,    73, 88888, 88888, 88888, 88888],
        [   86,    87,    80,    81, 88888, 88888, 88888, 88888],
        [   94,    95,    88,    89, 88888, 88888, 88888, 88888]], dtype=torch.int32)


    print(A)
    print(out.int())
    assert torch.equal(expected_out.int(), out.int())
    print('Hooooray')


def compile():
    ret = triton.compile(wrap_side_by_side_masked_loop, signature="*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32")
    print(ret.asm["ttir"])


test()
# compile()
