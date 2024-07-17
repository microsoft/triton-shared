import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl


# How do we handle cases where the strides change in each loop level?
@triton.jit
def nested2_strides_change(a_ptr, c_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = c_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)
        a_ptrs += 4 * (stride_n + 1)
        c_ptrs += 4 * (stride_n + 1)

        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += 4 * stride_n
            c_ptrs += 4 * stride_n

        a_ptrs += 4 * (stride_n + 2)
        c_ptrs += 4 * (stride_n + 2)


@triton.jit
def nested2_use_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)

        a_ptrs += 4 * stride_n
        c_ptrs += 4 * stride_n


        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += 4 * stride_n
            c_ptrs += 4 * stride_n

        a_ptrs += 4 * stride_n
        c_ptrs += 4 * stride_n


@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j in range(0, 2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k in range(0, 2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n

                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n


        a_ptrs += 2 * stride_n

def test1():
    n_rows = 4
    n_cols = 48
    expected = torch.tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  0.,  1.,  2.,  3.,  6.,  7.,  0.,  1.,
          8.,  9., 10., 11.,  0.,  1.,  8.,  9., 12., 13., 14., 15., 16., 17.,
         18., 19., 14., 15., 16., 17., 20., 21., 14., 15., 22., 23., 24., 25.,
         14., 15., 22., 23., 26., 27.],
        [48., 49., 50., 51., 52., 53., 48., 49., 50., 51., 54., 55., 48., 49.,
         56., 57., 58., 59., 48., 49., 56., 57., 60., 61., 62., 63., 64., 65.,
         66., 67., 62., 63., 64., 65., 68., 69., 62., 63., 70., 71., 72., 73.,
         62., 63., 70., 71., 74., 75.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.]], dtype=torch.int32, device='cpu')
    # x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.float32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)
    # print(x.stride(1))

    nested3[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    # ans = torch.sum(x, dim=1)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    # src = triton.compiler.ASTSource(
    #     fn=nested3,
    #     signature="*fp32,*fp32,i32,i32",
    # )
    # ret = triton.compile(
    #     src,
    # )
    # print(ret.asm["ttir"])
    # print(ret.asm["ttsharedir"])
    # print(ret.asm["llir"])
    # print(ret.asm["cpuasm"])
    # print('Pass')


def test2():
    n_rows = 4
    n_cols = 64
    expected = torch.tensor([[ 0,  1,  0,  0,  4,  5,  0,  0,  8,  9,  0,  0,  0,  0,  0,  0, 16, 17,
          0,  0, 20, 21,  0,  0, 24, 25,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [64, 65,  0,  0, 68, 69,  0,  0, 72, 73,  0,  0,  0,  0,  0,  0, 80, 81,
          0,  0, 84, 85,  0,  0, 88, 89,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], device='cpu',
       dtype=torch.int32)
    # x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)
    # print(x.stride(1))

    nested2_use_loop_results[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    # ans = torch.sum(x, dim=1)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")


def test3():
    n_rows = 4
    n_cols = 128
    expected = torch.tensor([[  0,   1,   0,   0,   0,   0,   0,   0,   8,   9,   0,   0,  12,  13,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          28,  29,   0,   0,   0,   0,   0,   0,  36,  37,   0,   0,  40,  41,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [128, 129,   0,   0,   0,   0,   0,   0, 136, 137,   0,   0, 140, 141,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         156, 157,   0,   0,   0,   0,   0,   0, 164, 165,   0,   0, 168, 169,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0]], device='cpu', dtype=torch.int32)
    # x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)
    # print(x.stride(1))

    nested2_strides_change[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    # ans = torch.sum(x, dim=1)
    # torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    # print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested2_strides_change,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')


test3()
