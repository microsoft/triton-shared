import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


def test_mask(device):
    @triton.jit
    def test(in0, out0):
        offs = 100 + tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        a = tl.load(in0 + offs, mask=offs < 4, other=-1)
        tl.store(out0 + out_offs, a)

    SIZE = 8
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -2, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    grid = lambda meta: (1,)

    src = triton.compiler.ASTSource(
        fn=test,
        signature={"in0": "*fp32", "out0": "*fp32"},
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

    print(output)
    test[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(output, torch.tensor([-1, -1, -1, -1, -2, -2, -2, -2], device=device, dtype=torch.int32))


def test_mask_with_scalar_in_conjunction(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    @triton.jit
    def kernel(in0, out0, mask, value):
        offs = tl.arange(0, 8)
        out_offs = tl.arange(0, 8)
        a = tl.load(in0 + offs, mask=(value < 5) & (offs < mask), other=-1)
        tl.store(out0 + out_offs, a)

    # Test scalar mask evaluate to True
    SIZE = 8
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -2, device=device, dtype=torch.int32)
    kernel[(1,)](input, output, 4, 3)
    torch.testing.assert_close(output, torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], device=device, dtype=torch.int32))

    # Test scalar mask evaluate to False
    SIZE = 8
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -2, device=device, dtype=torch.int32)
    kernel[(1,)](input, output, 4, 8)
    torch.testing.assert_close(output, torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1], device=device, dtype=torch.int32))