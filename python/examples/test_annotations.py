from __future__ import annotations
import torch
import triton
from triton.backends.triton_shared.driver import CPUDriver
import triton.language as tl
import pytest

triton.runtime.driver.set_active(CPUDriver())

def annotated_function(return_type=None, **arg_types):
    """A decorator to add annotations to a function."""

    def decorator(func):
        func.__annotations__ = {**arg_types, 'return': return_type}
        return func

    return decorator


# Test integer annotations
@pytest.mark.parametrize(("signed", "width"), [
    (signed, width) for signed in [False, True]\
                    for width in [8, 16, 32, 64]
] + [(False, 1)]
                         )
def test_int_annotation(signed, width):

    @triton.jit
    @annotated_function(X=torch.tensor, v=f"tl.{'' if signed else 'u'}int{width}")
    def _kernel(X, v):
        tl.store(X, v)

    h = _kernel[(1, )](torch.empty(1, device="cpu"), 3)
    pfx = 'si' if signed else 'ui'
    assert f'%arg1: i{width}' in h.asm["ttir"]
    assert f'arith.{pfx}tofp' in h.asm["ttir"]


# Test that unknown annotations do not emit an error
def test_unknown_annotation():

    @triton.jit
    def _kernel(X: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
        pass

    x = torch.empty(1, device="cpu")
    _kernel[(1, )](x, x.shape[0], 32)
    try:
        _kernel[(1, )](x.shape[0], x.shape[0], 32)
    except AttributeError:
        pass