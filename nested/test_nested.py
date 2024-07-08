import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl

# triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def nested1(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm,
                    stride_cn, BLOCK_SIZE_K: tl.constexpr):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                        offs_an[None, :] * stride_an)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)
        a_ptrs += BLOCK_SIZE_K * stride_an
        c_ptrs += BLOCK_SIZE_K * stride_an



# How do we handle cases where the strides change in each loop level?
@triton.jit
def nested_strides_change(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm,
                    stride_cn, BLOCK_SIZE_K: tl.constexpr):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                        offs_an[None, :] * stride_an)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)
        a_ptrs += BLOCK_SIZE_K * (stride_an + 1)
        c_ptrs += BLOCK_SIZE_K * (stride_an + 1)

        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an

@triton.jit
def nested2(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm,
                    stride_cn, BLOCK_SIZE_K: tl.constexpr):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                        offs_an[None, :] * stride_an)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)
        a_ptrs += BLOCK_SIZE_K * stride_an
        c_ptrs += BLOCK_SIZE_K * stride_an

        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an


@triton.jit
def nested3(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm,
                    stride_cn, BLOCK_SIZE_K: tl.constexpr):
    offs_am = tl.arange(0, 4)
    offs_an = tl.arange(0, 4)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                        offs_an[None, :] * stride_an)

    offs_cm = tl.arange(0, 4)
    offs_cn = tl.arange(0, 4)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
        None, :]

    for i in range(0, 2):

        a1 = tl.load(a_ptrs)
        tl.store(c_ptrs, a1)
        a_ptrs += BLOCK_SIZE_K * stride_an
        c_ptrs += BLOCK_SIZE_K * stride_an

        for j in range(0, 2):

            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a1)
            tl.store(c_ptrs, a2)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an


            for k in range(0, 2):
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                tl.store(c_ptrs, a2)
                tl.store(c_ptrs, a3)
                a_ptrs += BLOCK_SIZE_K * stride_an
                c_ptrs += BLOCK_SIZE_K * stride_an







def test():
    n_rows = 16
    n_cols = 32
    x = torch.rand([n_cols, n_rows], device="cpu", dtype=torch.float32)
    output = torch.empty([n_cols], device=x.device, dtype=x.dtype)
    BLOCK_SIZE = n_rows
    grid = lambda meta: (n_cols,)

    # reduce_kernel_2d[grid](x, output, x.stride(0), n_rows, BLOCK_SIZE=BLOCK_SIZE)
    # ans = torch.sum(x, dim=1)
    # torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)
    # print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested_strides_change,
        signature="*fp32,*fp32,i32,i32,i32,i32,i32,i32",
        constants={"BLOCK_SIZE_K": 32}
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    # print(ret.asm["ttsharedir"])
    # print(ret.asm["llir"])
    # print(ret.asm["cpuasm"])
    # print('Pass')

test()
