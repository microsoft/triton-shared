import pytest
import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver

@triton.jit
def gather_row(in0, out0, X:tl.constexpr=2, Y:tl.constexpr=4, Z:tl.constexpr=8, W:tl.constexpr=16):
    offs_2d_x = tl.arange(0, X * Y * Z)
    offs_2d_y = tl.arange(0, W)

    offs_4d_x = offs_2d_x[:, None] // (Y * Z)
    offs_4d_yz = offs_2d_x[:, None] % (Y * Z)

    offs_4d_y = offs_4d_yz // Z
    offs_4d_z = offs_4d_yz % Z

    offs_4d_w = offs_2d_y[None, :]

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w

    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)

def test_gather_row(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    
    r = gather_row[1,](input4D, output)
    torch.testing.assert_close(output, input)


@triton.jit
def gather_column(in0, out0, X:tl.constexpr=2, Y:tl.constexpr=4, Z:tl.constexpr=8, W:tl.constexpr=16):
    offs_2d_x = tl.arange(0, X)
    offs_2d_y = tl.arange(0, Y * Z * W)

    offs_4d_x = offs_2d_x[:, None]

    offs_4d_y = offs_2d_y[None, :] // (Z * W)

    offs_4d_zw = offs_2d_y[None, :] % (Z * W)

    offs_4d_z = offs_4d_zw // W
    offs_4d_w = offs_4d_zw % W

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w

    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)

def test_gather_column(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    
    r = gather_column[1,](input4D, output)
    torch.testing.assert_close(output, input)


@triton.jit
def gather_block(in0, out0, X:tl.constexpr=2, Y:tl.constexpr=4, Z:tl.constexpr=8, W:tl.constexpr=16):
    offs_3d_x = tl.arange(0, X)
    offs_3d_y = tl.arange(0, Y * Z)
    offs_3d_z = tl.arange(0, W)

    offs_4d_x = offs_3d_x[:, None, None]

    offs_4d_y = offs_3d_y[None, :, None] // Z

    offs_4d_z = offs_3d_y[None, :, None] % Z

    offs_4d_w = offs_3d_z[None, None, :]

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w

    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)

@pytest.mark.xfail(reason="result mismatch", run=False)
def test_gather_block(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    
    r = gather_block[1,](input4D, output)
    torch.testing.assert_close(output, input)

