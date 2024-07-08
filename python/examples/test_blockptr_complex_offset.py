import torch

import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl

triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def block_copy_kernel(a_ptr, b_ptr):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + 8,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=(0,))
    tl.store(b_block_ptr, a, boundary_check=(0,))



def test():
    input = torch.arange(0, 16, device="cpu", dtype=torch.float32)
    output = torch.full((4,), -1, device="cpu", dtype=torch.float32)
    expected = torch.arange(8, 12, device="cpu")
    grid = lambda meta: (1,)

    print("Output before: ", output)
    block_copy_kernel[grid](input, output)
    torch.equal(expected, output)
    print("Output after: ", output)
