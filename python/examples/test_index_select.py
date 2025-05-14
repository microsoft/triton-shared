import pytest
import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def index_select_row_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_I)
    row_indices = tl.load(indices + row_offsets * stride_i)

    col_offsets = tl.arange(0, BLOCK_N)
    input_pointers_0 = (
        input_ptr + row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
    )
    data = tl.load(input_pointers_0)

    tl.store(
        output_ptr
        + row_offsets[:, None] * o_stride_m
        + col_offsets[None, :] * o_stride_n,
        data,
    )


def index_select_row(input_tensor, indices, dim):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.empty(
        R, N, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    index_select_row_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_N=N,
    )
    return output_tensor


def test_index_select_row(device):
    M, N = 8, 16  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor(
        [1, 3, 5, 7], dtype=torch.int32, device=device
    )  # Example indices
    dim = 0  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = index_select_row(input_tensor, indices, dim)
    output_ref = torch.index_select(input_tensor, dim, indices)

    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def index_select_row_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_I)
    row_indices = tl.load(indices + row_offsets * stride_i)

    col_offsets = tl.arange(0, BLOCK_N)
    input_pointers_0 = (
        input_ptr + row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
    )
    data = tl.load(
        input_pointers_0, mask=col_offsets[None, :] < (BLOCK_N // 2), other=0
    )

    tl.store(
        output_ptr
        + row_offsets[:, None] * o_stride_m
        + col_offsets[None, :] * o_stride_n,
        data,
    )


def index_select_row_mask(input_tensor, indices, dim):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.empty(
        R, N, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    index_select_row_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_N=N,
    )
    return output_tensor


def test_index_select_row_mask(device):
    M, N = 8, 16  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor(
        [1, 3, 5, 7], dtype=torch.int32, device=device
    )  # Example indices
    dim = 0  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = index_select_row_mask(input_tensor, indices, dim)
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_ref[:, N // 2 :] = 0
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def index_select_col_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_I)
    col_indices = tl.load(indices + col_offsets)

    row_offsets = tl.arange(0, BLOCK_M)
    input_pointers_0 = (
        input_ptr + row_offsets[:, None] * stride_m + col_indices[None, :]
    )  # * stride_n
    data = tl.load(input_pointers_0)

    tl.store(
        output_ptr + row_offsets[:, None] * o_stride_m + col_offsets[None, :], data
    )


def index_select_col(input_tensor, indices, dim):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.full(
        (M, R), -1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    print(stride_i, stride_m, stride_n, o_stride_m, o_stride_n)
    index_select_col_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_M=M,
    )
    return output_tensor


def test_index_select_col(device):
    M, N = 8, 8  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)  # Example indices
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_triton = index_select_col(input_tensor, indices, dim)
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def index_select_col_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_I)
    col_indices = tl.load(indices + col_offsets)

    row_offsets = tl.arange(0, BLOCK_M)
    input_pointers_0 = (
        input_ptr + row_offsets[:, None] * stride_m + col_indices[None, :]
    )  # * stride_n
    data = tl.load(
        input_pointers_0, mask=row_offsets[:, None] < (BLOCK_M // 2), other=0
    )

    tl.store(
        output_ptr + row_offsets[:, None] * o_stride_m + col_offsets[None, :], data
    )


def index_select_col_mask(input_tensor, indices, dim):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.full(
        (M, R), -1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    print(stride_i, stride_m, stride_n, o_stride_m, o_stride_n)
    index_select_col_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_M=M,
    )
    return output_tensor


def test_index_select_col_mask(device):
    M, N = 8, 8  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)  # Example indices
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_ref[N // 2 :, :] = 0
    output_triton = index_select_col_mask(input_tensor, indices, dim)
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def index_select_3d_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    stride_k,
    o_stride_m,
    o_stride_n,
    o_stride_k,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_offsets = tl.arange(0, BLOCK_I)
    n_indices = tl.load(indices + n_offsets)

    m_offsets = tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    input_offsets = (
        m_offsets[:, None, None] * stride_m
        + n_indices[None, :, None] * stride_n
        + k_offsets[None, None, :] * stride_k
    )

    input_pointers_0 = input_ptr + input_offsets
    data = tl.load(input_pointers_0)

    out_offsets = (
        m_offsets[:, None, None] * o_stride_m
        + n_offsets[None, :, None] * o_stride_n
        + k_offsets[None, None, :] * o_stride_k
    )
    tl.store(output_ptr + out_offsets, data)


def index_select_3d(input_tensor, indices, dim):
    M, N, K = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.full(
        (M, R, K), -1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    stride_k = input_tensor.stride(2)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    o_stride_k = output_tensor.stride(2)
    index_select_3d_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        stride_k,
        o_stride_m,
        o_stride_n,
        o_stride_k,
        BLOCK_I=R,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return output_tensor


def test_index_select_3d(device):
    M, N, K = 4, 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, K, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)  # Example indices
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_triton = index_select_3d(input_tensor, indices, dim)
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def index_select_3d_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    stride_k,
    o_stride_m,
    o_stride_n,
    o_stride_k,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_offsets = tl.arange(0, BLOCK_I)
    n_indices = tl.load(indices + n_offsets)

    m_offsets = tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    input_offsets = (
        m_offsets[:, None, None] * stride_m
        + n_indices[None, :, None] * stride_n
        + k_offsets[None, None, :] * stride_k
    )

    input_pointers_0 = input_ptr + input_offsets
    data = tl.load(
        input_pointers_0,
        mask=m_offsets[:, None, None] < (BLOCK_M // 2)
        and k_offsets[None, None, :] < (BLOCK_K // 2),
        other=0,
    )

    out_offsets = (
        m_offsets[:, None, None] * o_stride_m
        + n_offsets[None, :, None] * o_stride_n
        + k_offsets[None, None, :] * o_stride_k
    )
    tl.store(output_ptr + out_offsets, data)


def index_select_3d_mask(input_tensor, indices, dim):
    M, N, K = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.full(
        (M, R, K), -1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    stride_k = input_tensor.stride(2)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    o_stride_k = output_tensor.stride(2)
    index_select_3d_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        stride_k,
        o_stride_m,
        o_stride_n,
        o_stride_k,
        BLOCK_I=R,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return output_tensor


def test_index_select_3d_mask(device):
    M, N, K = 4, 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, K, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)  # Example indices
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_ref[M // 2 :, :, :] = 0
    output_ref[:, :, K // 2 :] = 0
    output_triton = index_select_3d_mask(input_tensor, indices, dim)
    print("output_triton", output_triton.cpu())
    print("output_ref", output_ref.cpu())
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_row_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_I)
    row_indices = tl.load(indices + row_offsets * stride_i)

    col_offsets = tl.arange(0, BLOCK_N)
    input_pointers_0 = (
        input_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :] * stride_n
    )
    data = tl.load(input_pointers_0)

    tl.store(
        output_ptr
        + row_indices[:, None] * o_stride_m
        + col_offsets[None, :] * o_stride_n,
        data,
    )


def scatter_row(dst, dim, indices, input_tensor):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    scatter_row_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_N=N,
    )
    return output_tensor


def test_scatter_row(device):
    M, N = 8, 8  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor(
        [1, 3, 5, 7], dtype=torch.int64, device=device
    )  # Example indices
    dst = torch.zeros((8, 16), dtype=input_tensor.dtype, device=device)

    dim = 0  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = scatter_row(dst, dim, indices, input_tensor)

    # copy all columns of input_tensor.
    row_indices = indices.reshape(4, 1).repeat(1, N)

    output_ref = torch.scatter(dst, dim, row_indices, input_tensor)

    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_row_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_I)
    row_indices = tl.load(indices + row_offsets * stride_i)

    col_offsets = tl.arange(0, BLOCK_N)
    input_pointers_0 = (
        input_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :] * stride_n
    )
    data = tl.load(input_pointers_0)

    tl.store(
        output_ptr
        + row_indices[:, None] * o_stride_m
        + col_offsets[None, :] * o_stride_n,
        data,
        mask=col_offsets[None, :] < (BLOCK_N // 2),
    )


def scatter_row_mask(dst, dim, indices, input_tensor):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    scatter_row_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_N=N,
    )
    return output_tensor


def test_scatter_row_mask(device):
    M, N = 8, 8  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor(
        [1, 3, 5, 7], dtype=torch.int64, device=device
    )  # Example indices
    dst = torch.zeros((8, 16), dtype=input_tensor.dtype, device=device)

    dim = 0  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = scatter_row_mask(dst, dim, indices, input_tensor)

    # copy all columns of input_tensor.
    row_indices = indices.reshape(4, 1).repeat(1, N)

    output_ref = torch.scatter(dst, dim, row_indices, input_tensor)
    output_ref[:, N // 2 :] = 0
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_col_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_I)
    col_indices = tl.load(indices + col_offsets * 1)

    row_offsets = tl.arange(0, BLOCK_M)
    input_pointers_0 = input_ptr + row_offsets[:, None] * 4 + col_offsets[None, :] * 1
    data = tl.load(input_pointers_0)

    tl.store(output_ptr + row_offsets[:, None] * 4 + col_indices[None, :] * 1, data)


def scatter_col(dst, dim, indices, input_tensor):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    scatter_col_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_M=M,
    )
    return output_tensor


def test_scatter_col(device):
    M, N = 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int64, device=device)  # Example indices
    dst = torch.zeros((4, 4), dtype=input_tensor.dtype, device=device)

    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = scatter_col(dst, dim, indices, input_tensor)

    # copy all columns of input_tensor.
    col_indices = indices.repeat(4, 1)

    output_ref = torch.scatter(dst, dim, col_indices, input_tensor)
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_col_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    o_stride_m,
    o_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_I)
    col_indices = tl.load(indices + col_offsets * 1)

    row_offsets = tl.arange(0, BLOCK_M)
    input_pointers_0 = input_ptr + row_offsets[:, None] * 4 + col_offsets[None, :] * 1
    data = tl.load(input_pointers_0)

    tl.store(
        output_ptr + row_offsets[:, None] * 4 + col_indices[None, :] * 1,
        data,
        mask=row_offsets[:, None] < (BLOCK_M // 2),
    )


def scatter_col_mask(dst, dim, indices, input_tensor):
    M, N = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)

    scatter_col_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_I=R,
        BLOCK_M=M,
    )
    return output_tensor


def test_scatter_col_mask(device):
    M, N = 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int64, device=device)  # Example indices
    dst = torch.zeros((4, 4), dtype=input_tensor.dtype, device=device)

    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_triton = scatter_col_mask(dst, dim, indices, input_tensor)

    # copy all columns of input_tensor.
    col_indices = indices.repeat(4, 1)

    output_ref = torch.scatter(dst, dim, col_indices, input_tensor)
    output_ref[M // 2 :, :] = 0
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_3d_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    stride_k,
    o_stride_m,
    o_stride_n,
    o_stride_k,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_offsets = tl.arange(0, BLOCK_I)
    n_indices = tl.load(indices + n_offsets)

    m_offsets = tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    input_offsets = (
        m_offsets[:, None, None] * stride_m
        + n_offsets[None, :, None] * stride_n
        + k_offsets[None, None, :] * stride_k
    )

    input_pointers_0 = input_ptr + input_offsets
    data = tl.load(input_pointers_0)

    out_offsets = (
        m_offsets[:, None, None] * o_stride_m
        + n_indices[None, :, None] * o_stride_n
        + k_offsets[None, None, :] * o_stride_k
    )
    tl.store(output_ptr + out_offsets, data)


def scatter_3d(dst, dim, indices, input_tensor):
    M, N, K = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()

    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    stride_k = input_tensor.stride(2)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    o_stride_k = output_tensor.stride(2)
    print(stride_i, stride_m, stride_n, stride_k, o_stride_m, o_stride_n, o_stride_k)
    scatter_3d_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        stride_k,
        o_stride_m,
        o_stride_n,
        o_stride_k,
        BLOCK_I=R,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return output_tensor


def test_scatter_3d(device):
    M, N, K = 4, 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, K, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int64, device=device)  # Example indices
    dst = torch.zeros((4, 8, 4), dtype=input_tensor.dtype, device=device)
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    print("input_tensor", input_tensor.cpu())
    col_indices = indices.reshape(2, 1).repeat(M, 1, K)
    output_ref = torch.scatter(dst, dim, col_indices, input_tensor)
    print("output_ref", output_ref.cpu())
    output_triton = scatter_3d(dst, dim, indices, input_tensor)
    print("output_triton", output_triton.cpu())
    torch.testing.assert_close(output_triton, output_ref)


@triton.jit
def scatter_3d_mask_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    stride_k,
    o_stride_m,
    o_stride_n,
    o_stride_k,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_offsets = tl.arange(0, BLOCK_I)
    n_indices = tl.load(indices + n_offsets)

    m_offsets = tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    input_offsets = (
        m_offsets[:, None, None] * stride_m
        + n_offsets[None, :, None] * stride_n
        + k_offsets[None, None, :] * stride_k
    )

    input_pointers_0 = input_ptr + input_offsets
    data = tl.load(input_pointers_0)

    out_offsets = (
        m_offsets[:, None, None] * o_stride_m
        + n_indices[None, :, None] * o_stride_n
        + k_offsets[None, None, :] * o_stride_k
    )
    tl.store(
        output_ptr + out_offsets,
        data,
        mask=m_offsets[:, None, None] < (BLOCK_M // 2)
        and k_offsets[None, None, :] < (BLOCK_K // 2),
    )


def scatter_3d_mask(dst, dim, indices, input_tensor):
    M, N, K = input_tensor.shape
    R = indices.shape[0]
    output_tensor = dst.clone()

    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    stride_k = input_tensor.stride(2)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    o_stride_k = output_tensor.stride(2)
    print(stride_i, stride_m, stride_n, stride_k, o_stride_m, o_stride_n, o_stride_k)
    scatter_3d_mask_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        stride_k,
        o_stride_m,
        o_stride_n,
        o_stride_k,
        BLOCK_I=R,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return output_tensor


def test_scatter_3d_mask(device):
    M, N, K = 4, 4, 4  # Example sizes
    input_tensor = torch.randn(M, N, K, device=device)  # Random tensor
    indices = torch.tensor([1, 3], dtype=torch.int64, device=device)  # Example indices
    dst = torch.zeros((4, 8, 4), dtype=input_tensor.dtype, device=device)
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    print("input_tensor", input_tensor.cpu())
    col_indices = indices.reshape(2, 1).repeat(M, 1, K)
    output_ref = torch.scatter(dst, dim, col_indices, input_tensor)
    output_ref[M // 2 :, :, :] = 0
    output_ref[:, :, K // 2 :] = 0
    print("output_ref", output_ref.cpu())
    output_triton = scatter_3d_mask(dst, dim, indices, input_tensor)
    print("output_triton", output_triton.cpu())
    torch.testing.assert_close(output_triton, output_ref)
