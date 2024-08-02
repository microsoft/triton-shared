# This test is one of the tests from triton/python/test/unit/language/test_core.py
# with an addition of CPUDriver activation
import torch

import numpy as np
import pytest
from typing import Optional, Union
from numpy.random import RandomState

import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl
from triton.runtime.jit import TensorWrapper, reinterpret

triton.runtime.driver.set_active(CPUDriver())

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']


def to_triton(x: np.ndarray, device, dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Workaround. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32') & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')

@pytest.mark.parametrize("dtype_str", ['int32'])
def test_umulhi(dtype_str, device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.umulhi(x, y)
        tl.store(Z + tl.arange(0, N), z)

    def umulhi32(a, b):
        # Convert to 64-bit unsigned integers to prevent overflow
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)

        # Perform the multiplication in 64-bit
        product_64 = a_64 * b_64

        # Shift right by 32 bits to get the high part of the product
        result_high_32 = product_64 >> 32
        return result_high_32

    rs = RandomState(17)
    N = 128
    x = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    x_tri = to_triton(x, device=device)
    y = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    y_tri = to_triton(y, device=device)
    z_tri = torch.zeros_like(x_tri)
    kernel[(1, )](x_tri, y_tri, z_tri, N=N)

    z_ref = umulhi32(x, y)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))