import torch
import triton
import triton.language as tl
import sys

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

# Buffer overflow at line 28

@triton.jit
def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask is erroneously set to true
    # no longer guards against out of bounds accesses if n_elements % BLOCK_SIZE != 0
    # correct: mask = offsets < n_elements
    mask = tl.full((BLOCK_SIZE, ), 1, dtype=tl.int1)

    # store BLOCK_SIZE 1s into the output_ptr
    output = tl.full((BLOCK_SIZE, ), 1, dtype=tl.float16)
    
    # write output to the output_ptr
    # one write will be a buffer overflow
    tl.store(output_ptr + offsets, output, mask=mask)

size = 2049

output = torch.empty((size, )).to("cpu")
print(output)

grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
kernel[grid](output, size, BLOCK_SIZE=2048)

print(output)
