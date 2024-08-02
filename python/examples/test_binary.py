# This test is one of the tests from triton/python/test/unit/language/test_core.py
# with an addition of CPUDriver activation
import itertools
import re
from typing import Optional, Union
import textwrap

import numpy as np
import pytest
import torch
import inspect
from numpy.random import RandomState

import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl
from triton.runtime.jit import TensorWrapper, reinterpret

triton.runtime.driver.set_active(CPUDriver())


int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_float8_dtypes = ['float8_e4m3fn', 'float8_e5m2']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']

# TODO: enable multiple cta cluster testing.
# num_ctas_list = [1, 4] if torch.cuda.get_device_capability()[0] == 9 else [1]
num_ctas_list = [1]

GPU_DIALECT = "triton_gpu"
THREADS_PER_WARP = 1

def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


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


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel



def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', num_ctas=1,
                 y_low=None, y_high=None, test_broadcast=True):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    check_type_supported(dtype_y, device)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_lhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_rhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    replacements = {'GENERATE_TEST_HERE': expr}
    kernel = patch_kernel(kernel, replacements)
    kernel_broadcast_lhs = patch_kernel(kernel_broadcast_lhs, replacements)
    kernel_broadcast_rhs = patch_kernel(kernel_broadcast_rhs, replacements)

    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')

    def do_test(x, y, kernel_fn):
        # reference result
        z_ref = eval(expr if numpy_expr is None else numpy_expr)
        dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
        if dtype_z is not None:
            z_ref = z_ref.astype(dtype_z)
        # triton result
        x_tri = to_triton(x, device=device, dst_type=dtype_x)
        y_tri = to_triton(y, device=device, dst_type=dtype_y)
        z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
        kernel_fn[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
        err_msg = f"{expr}, {kernel_fn.__name__}"
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=err_msg, atol=1e-3, rtol=0.01)

    do_test(x, y, kernel)
    if test_broadcast:
        do_test(x[:1].reshape(()), y, kernel_broadcast_lhs)
        do_test(x, y[:1].reshape(()), kernel_broadcast_rhs)


def _mod_operation_ill_conditioned(dtype_x, dtype_y) -> bool:
    # The result of x % y is ill-conditioned if x % y is much smaller than x.
    # pytorch/CUDA has slightly different (probably better) rounding on
    # remainders than stock LLVM. We currently don't expect to match it
    # bit-for-bit.
    return (dtype_x, dtype_y) in [
        ('int32', 'bfloat16'),
        ('int32', 'float16'),
        ('int32', 'float32'),
        ('int64', 'bfloat16'),
        ('int64', 'float16'),
        ('int64', 'float32'),
        ('int64', 'float64'),
        ('uint16', 'bfloat16'),
        ('uint16', 'float16'),
        ('uint16', 'float32'),
        ('uint32', 'bfloat16'),
        ('uint32', 'float16'),
        ('uint32', 'float32'),
        ('uint64', 'bfloat16'),
        ('uint64', 'float16'),
        ('uint64', 'float32'),
        ('uint64', 'float64'),
    ]


def test_dtype_codegen():
    for dtype in dtypes_with_bfloat16:
        full_name = f"triton.language.{dtype}"
        assert repr(eval(full_name)) == full_name


# ---------------
# test binary ops
# ---------------

@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['+', '-', '*', '/', '%']
    for dtype_x in dtypes
    for dtype_y in dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bin_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f' x {op} y'
    if op == '%' and dtype_x in int_dtypes + uint_dtypes and dtype_y in int_dtypes + uint_dtypes:
        # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
        numpy_expr = 'np.fmod(x, y)'
    elif op in ('/', '%') and dtype_x in ('int16', 'float16', 'bfloat16') and dtype_y in ('int16', 'float16',
                                                                                          'bfloat16'):
        # Triton promotes 16-bit floating-point / and % to 32-bit because there
        # are no native div or FRem operations on float16. Since we have to
        # convert anyway, we may as well take the accuracy bump.
        numpy_expr = f'x.astype(np.float32) {op} y.astype(np.float32)'
    elif (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if op == '%' and _mod_operation_ill_conditioned(dtype_x, dtype_y):
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
    elif (op in ('%', '/') and ((dtype_x in int_dtypes and dtype_y in uint_dtypes) or
                                (dtype_x in uint_dtypes and dtype_y in int_dtypes))):
        with pytest.raises(triton.TritonError, match='Cannot use .* because they have different signedness'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
    else:
        _test_binary(
            dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas,
            # fails with values where fmod(x, y) is roughly zero, but happens to
            # pass with the random values chosen for non-broadcast tests
            test_broadcast=(op != "%"))
