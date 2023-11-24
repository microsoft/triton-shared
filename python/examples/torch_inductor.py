import triton
import triton.language as tl

import torch

@triton.jit
def triton_(in_ptr2, out_ptr2, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x0 = xindex
    roffset = 0
    rindex = roffset + rbase
    rmask = rindex < rnumel
    r2 = rindex
    tmp3 = tl.load(in_ptr2 + (r2 + (xnumel*x0)), rmask, other=77)
    tl.store(out_ptr2 + (XBLOCK * tl.arange(0, RBLOCK)[None, :] + tl.arange(0, XBLOCK)[:, None]), tmp3)


@triton.jit
def triton__(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + (XBLOCK * tl.arange(0, RBLOCK)[None, :] + tl.arange(0, XBLOCK)[:, None]) % 256
    tmp3 = tl.load(in_ptr2 + xindex)
    tl.store(out_ptr2 + (XBLOCK * tl.arange(0, RBLOCK)[None, :] + tl.arange(0, XBLOCK)[:, None]), tmp3)



def test():
    torch.set_printoptions(threshold=10_000)

    device = "cpu"
    xnumel = 128
    rnumel = 32

    XBLOCK = 4
    RBLOCK = 64
    A = torch.arange(0, xnumel * rnumel, device=device, dtype=torch.int32).reshape((xnumel, rnumel))
    out = torch.full((XBLOCK, RBLOCK), 88888, device=device, dtype=torch.int32)
    grid = lambda meta: (1,)

    triton_[grid](
        A,
        out,
        rnumel,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK
    )

    expected_out = torch.tensor([[  0, 128, 256, 384,   1, 129, 257, 385,   2, 130, 258, 386,   3, 131,
         259, 387,   4, 132, 260, 388,   5, 133, 261, 389,   6, 134, 262, 390,
           7, 135, 263, 391,   8, 136, 264, 392,   9, 137, 265, 393,  10, 138,
         266, 394,  11, 139, 267, 395,  12, 140, 268, 396,  13, 141, 269, 397,
          14, 142, 270, 398,  15, 143, 271, 399],
        [ 16, 144, 272, 400,  17, 145, 273, 401,  18, 146, 274, 402,  19, 147,
         275, 403,  20, 148, 276, 404,  21, 149, 277, 405,  22, 150, 278, 406,
          23, 151, 279, 407,  24, 152, 280, 408,  25, 153, 281, 409,  26, 154,
         282, 410,  27, 155, 283, 411,  28, 156, 284, 412,  29, 157, 285, 413,
          30, 158, 286, 414,  31, 159, 287, 415],
        [ 77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77],
        [ 77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,
          77,  77,  77,  77,  77,  77,  77,  77]], device=device,
       dtype=torch.int32)

    print(out)
    assert torch.equal(expected_out.int(), out.int())
    print('Passed')

def compile():
    ret = triton.compile(triton_, signature="*i32,*i32,i32,i32,i32", constants={"XBLOCK": 64, "RBLOCK": 64})
    print(ret.asm["ttir"])


test()
