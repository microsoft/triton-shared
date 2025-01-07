import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver

def test_gather_div(device):

    @triton.jit
    def gather_simple_no_mask(in0, out0):
        offs = tl.arange(0, 64)
        out_offs = tl.arange(0, 64)
        for i in range(0, 2):
            offs = offs // 10 + (i + 1 * 5) % 64
            a = tl.load(in0 + offs)
            tl.store(out0 + out_offs, a)
            offs += 64
            out_offs += 64


    @triton.jit
    def gather_simple_mask_no_other(in0, out0):
        offs = tl.arange(0, 64)
        out_offs = tl.arange(0, 64)
        mask_bound = 8
        for i in range(0, 2):
            gather_offs = offs // 4
            a = tl.load(in0 + gather_offs, mask=gather_offs < mask_bound)
            tl.store(out0 + out_offs, a)
            mask_bound += 16
            offs += 64
            out_offs += 64


    @triton.jit
    def gather_simple_mask_with_other(in0, out0):
        offs = tl.arange(0, 64)
        out_offs = tl.arange(0, 64)
        mask_bound = 8
        for i in range(0, 2):
            gather_offs = offs // 4
            a = tl.load(in0 + gather_offs, mask=gather_offs < mask_bound, other=-1)
            tl.store(out0 + out_offs, a)
            mask_bound += 16
            offs += 64
            out_offs += 64


    @triton.jit
    def masked_gather_scatter(in0, out0):
        offs = tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        for i in range(0, 2):
            # offs = offs % i
            offs = offs // 3 + i
            mask = offs < 64
            a = tl.load(in0 + offs, mask=mask, other=99)
            tl.store(out0 + offs, a, mask=mask)
            offs += 4
            out_offs += 4

    @triton.jit
    def gather(in0, out0):
        offs = tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        for i in range(0, 2):
            # offs = offs % i
            offs = offs // 3 + i
            mask = offs < 64
            a = tl.load(in0 + offs, mask=mask)
            tl.store(out0 + offs, a)
            offs += 4
            out_offs += 4

            for j in range(0, 2):
                offs = offs // ((i + 1) * (j + 1)) + i
                mask = offs < 64
                a = tl.load(in0 + offs, mask=mask)
                tl.store(out0 + offs, a)
                offs += 4
                out_offs += 4

            # offs = offs % i
            offs = offs // 3 + i
            mask = offs < 64
            a = tl.load(in0 + offs, mask=mask)
            tl.store(out0 + offs, a)
            offs += 4
            out_offs += 4

    @triton.jit
    def mask_off(in0, out0):
        offs = tl.arange(0, 16)
        a = tl.load(in0 + offs, mask=offs < 0)
        tl.store(out0 + offs, a)


    SIZE = 128
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    grid = lambda meta: (1,)

    # print(output)
    # gather_simple_mask[grid](input, output)
    # mask_off[grid](input, output)
    # print(input)
    # print(output)
    src = triton.compiler.ASTSource(
        fn=masked_gather_scatter,
        signature="*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])


test_gather_div('cuda')
