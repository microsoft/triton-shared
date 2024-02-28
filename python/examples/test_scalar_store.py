import torch

import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def reduce_kernel_2d(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    base_ptr = output_ptr + pid0
    for i in range(0, BLOCK_SIZE):
        output = i
        tl.store(base_ptr, output)
        base_ptr += 1


def test():
    BLOCK_SIZE = 8
    x = torch.full([BLOCK_SIZE], -1, device="cpu", dtype=torch.float32)
    output = torch.full((BLOCK_SIZE,), -99, device=x.device, dtype=x.dtype)
    grid = lambda meta: (1,)

    reduce_kernel_2d[grid](output, BLOCK_SIZE=BLOCK_SIZE)
    ans = torch.arange(BLOCK_SIZE, device="cpu", dtype=torch.float32)
    print('Expected: ', ans)
    print('Actual: ', output)
    torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)
    print("Pass!")
