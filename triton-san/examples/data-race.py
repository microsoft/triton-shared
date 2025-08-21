import torch
import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

# Data race pair: any two program ids at line 28

@triton.jit
def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # block_start is erroneously set to 0, independent of pid
    # this will cause both programs to access and write the same elements
    # correct: block_start = pid * BLOCK_SIZE
    block_start = 0

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # store BLOCK_SIZE values containing pid into the output_ptr
    # this is so that the values stored by the program are different
    output = tl.full((BLOCK_SIZE, ), pid, dtype=tl.float16)
    
    # the race will occur between the programs here
    tl.store(output_ptr + offsets, output, mask=mask)

size = 256

output = torch.empty((size, )).to("cpu") # initialize GPU tensor with one extra element

grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
kernel[grid](output, size, BLOCK_SIZE=2)