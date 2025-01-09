import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver

@triton.jit
def gather_simple(in0, in1, out0):
    offs = tl.arange(0, 8)
    in0_ptrs = in0 + offs
    in1_ptrs = in1 + offs
    ptrs = tl.cat(in0_ptrs, in1_ptrs, can_reorder=True)
    c = tl.load(ptrs)
    # a = tl.load(in0_ptrs)
    # b = tl.load(in1_ptrs)
    # c = tl.cat(a, b, can_reorder=True)
    out_offs = tl.arange(0, 16)
    tl.store(out0 + out_offs, c)


def compile(device):
    src = triton.compiler.ASTSource(
        fn=gather_simple,
        signature="*i32,*i32,*i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

def run_test(triton_kernel, device, expected_output):
    SIZE = 16
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input1 = torch.arange(SIZE, SIZE + SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    grid = lambda meta: (1,)

    print(output)
    triton_kernel[grid](input, input1, output)
    print(input)
    print(output)
    # torch.testing.assert_close(output, expected_output)


def test_gather_multiple_bases(device):
    expected_output = torch.tensor([ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,
         8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15], device=device, dtype=torch.int32)
    run_test(gather_simple, device, expected_output)

test_gather_multiple_bases('cpu')
# compile('cpu')
