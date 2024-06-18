import torch

import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def addptr(in0, out0):
    for i in range(0, 10, 2):
        in1 = in0 + 1 + i
        in2 = in1 + 1

        out1 = out0 + 1 + i
        out2 = out1 + 1

        a1 = tl.load(in1)
        a2 = tl.load(in2)

        tl.store(out1, a1)
        tl.store(out2, a2)



def test():
    input = torch.arange(0, 11, device="cpu", dtype=torch.float32)
    output = torch.full((11,), 0, device="cpu", dtype=torch.float32)
    grid = lambda meta: (1,)

    print(output)
    addptr[grid](input, output)
    print(input)
    print(output)
    assert torch.equal(input, output)

    src = triton.compiler.ASTSource(
        fn=addptr,
        signature="*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')
