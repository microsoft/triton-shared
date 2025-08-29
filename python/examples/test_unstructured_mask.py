import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
from triton.backends.triton_shared.compiler import _get_triton_shared_opt_path

import os
from pathlib import Path
import subprocess
import tempfile

def run_triton_to_structured(ttir_code):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path()

        subprocess_args = [triton_shared_opt_path, src_path, "--triton-to-structured", "--remove-dead-values", "-o", dst_path]

        subprocess.check_call(subprocess_args)
        return Path(dst_path).read_text()

@triton.jit
def unstructured_mask_2d_kernel(in_ptr, out_ptr, mask_m_ptr, mask_n_ptr, m, n, M:tl.constexpr, N:tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    mask_m = tl.load(mask_m_ptr + offs_m, mask=offs_m < m, other=0) != 0
    mask_n = tl.load(mask_n_ptr + offs_n, mask=offs_n < n, other=0) != 0

    in_ptrs = in_ptr + offs_m[:, None] * N + offs_n[None, :]
    # dim 0 with unstructured mask.
    v = tl.load(in_ptrs, mask=mask_m[:, None] and offs_n[None, :] < n, other=-2)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    # dim 1 with unstructured mask.
    tl.store(out_ptrs, v, mask=offs_m[:, None] < m and mask_n[None, :])

def test_unstructured_mask_2d(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    m = 6
    n = 16
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (m * n), device=device, dtype=torch.float32).reshape(m, n)
    output = torch.full_like(input, -1)
    mask_m = torch.tensor([1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)
    
    mask_n = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1,
                            1, 0, 1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)

    res = unstructured_mask_2d_kernel[1,](input, output, mask_m.to(torch.int8), mask_n.to(torch.int8), m, n, M, N)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 2, "Expected exactly two calls to tts.make_gather_scatter_tptr"

    v = torch.full_like(input, -2)
    v[mask_m,:] = input[mask_m,:]

    expected_output = torch.full_like(output, -1)
    expected_output[:,mask_n] = v[:,mask_n]

    torch.testing.assert_close(output, expected_output)


@triton.jit
def unstructured_mask_3d_kernel(in_ptr, out_ptr, mask_m_ptr, mask_n_ptr, b, m, n, stride_b, stride_m, stride_n, B: tl.constexpr, M:tl.constexpr, N:tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_b = tl.arange(0, B)

    mask_m = tl.load(mask_m_ptr + offs_m, mask=offs_m < m, other=0) != 0
    mask_n = tl.load(mask_n_ptr + offs_n, mask=offs_n < n, other=0) != 0

    in_ptrs = in_ptr + offs_b[:, None, None] * stride_b + offs_m[None, :, None] * stride_m + offs_n[None, None, :] * stride_n
    # dim 1 with unstructured mask.
    v = tl.load(in_ptrs, mask= offs_b[:, None, None] < b and mask_m[None, :, None] and offs_n[None, None, :] < n, other=-2)
    out_ptrs = out_ptr + offs_b[:, None, None] * stride_b + offs_m[None, :, None] * stride_m + offs_n[None, None, :] * stride_n
    # dim 2 with unstructured mask.
    tl.store(out_ptrs, v, mask= offs_b[:, None, None] < b and offs_m[None, :, None] < m and mask_n[None, None, :])

def test_unstructured_mask_3d(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    b = 4
    m = 6
    n = 16
    B = triton.next_power_of_2(b)
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (b * m * n), device=device, dtype=torch.float32).reshape(b, m, n)
    output = torch.full_like(input, -1)
    mask_m = torch.tensor([1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)

    mask_n = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1,
                            1, 0, 1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)

    stride_b = input.stride(0)
    stride_m = input.stride(1)
    stride_n = input.stride(2)
    res = unstructured_mask_3d_kernel[1,](input, output, mask_m.to(torch.int8), mask_n.to(torch.int8), b, m, n,
                                    stride_b, stride_m, stride_n,
                                    B, M, N)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 2, "Expected exactly two calls to tts.make_gather_scatter_tptr"

    v = torch.full_like(input, -2)
    v[:, mask_m,:] = input[:, mask_m,:]

    expected_output = torch.full_like(output, -1)
    expected_output[:, :, mask_n] = v[:, :, mask_n]

    torch.testing.assert_close(output, expected_output)

# non-continuous ld/st and (offs_n < n)[:, None] pattern.

@triton.jit
def unstructured_mask_2d_non_continuous_load_kernel(in_ptr, out_ptr, index_m_ptr, im, I_M:tl.constexpr, N:tl.constexpr, m, n, stride_m, stride_n):
    offs_m = tl.arange(0, I_M)
    offs_n = tl.arange(0, N)

    index_m = tl.load(index_m_ptr + offs_m, mask=offs_m < im, other=0)
    mask_m_i = index_m < m and offs_m < im


    # dim 0 with unstructured offset.
    in_ptrs = in_ptr + index_m[:, None] * stride_m + offs_n[None, :] * stride_n
    # dim 0 with unstructured mask.
    v = tl.load(in_ptrs, mask=mask_m_i[:, None] and offs_n[None, :] < n, other=-2)

    out_ptrs = out_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(out_ptrs, v, mask=offs_m[:, None] < im and offs_n[None, :] < n)

def test_unstructured_mask_2d_non_continuous_load(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    m = 6
    n = 16
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (m * n), device=device, dtype=torch.float32).reshape(m, n)
    index_m = torch.tensor([1, 3, 7], device=device, dtype=torch.int32)

    index_n = torch.tensor([10, 20, 15], device=device, dtype=torch.int32)

    stride_m = input.stride(0)
    stride_n = input.stride(1)
    I_M = triton.next_power_of_2(len(index_m))
    I_N = triton.next_power_of_2(len(index_n))

    output = torch.full((len(index_m), n), -1, device=device, dtype=torch.float32)
    res = unstructured_mask_2d_non_continuous_load_kernel[1,](input, output, index_m, len(index_m), I_M, N, m, n, stride_m, stride_n)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 1, "Expected exactly one call to tts.make_gather_scatter_tptr"

    expected_output = torch.full((len(index_m), n), -2, device=device, dtype=torch.float32)
    mask_m = index_m < m
    index_m = index_m[mask_m]
    expected_output[:len(index_m),:] = input[index_m,:]

    torch.testing.assert_close(output, expected_output)


@triton.jit
def unstructured_mask_2d_non_continuous_store_kernel(in_ptr, out_ptr, index_n_ptr, i_n, I_N:tl.constexpr, M:tl.constexpr, m, n, stride_m, stride_n):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, I_N)

    index_n = tl.load(index_n_ptr + offs_n, mask=offs_n < n, other=0)
    mask_n_i = index_n < n and offs_n < i_n


    in_ptrs = in_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    v = tl.load(in_ptrs, mask=offs_m[:, None] < m and offs_n[None, :] < i_n, other=-2)
    # dim 1 with unstructured offset.
    out_ptrs = out_ptr + offs_m[:, None] * stride_m + index_n[None, :] * stride_n
    # dim 1 with unstructured mask.
    tl.store(out_ptrs, v, mask=offs_m[:, None] < m and mask_n_i[None, :])

def test_unstructured_mask_2d_non_continuous_store(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    m = 6
    n = 16
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (m * n), device=device, dtype=torch.float32).reshape(m, n)
 
    index_n = torch.tensor([10, 20, 15, 1, 3, 5, 7, 0, 2, 4, 6, 8], device=device, dtype=torch.int32)

    stride_m = input.stride(0)
    stride_n = input.stride(1)

    I_N = triton.next_power_of_2(len(index_n))

    output = torch.full((m, n), -1, device=device, dtype=torch.float32)
    res = unstructured_mask_2d_non_continuous_store_kernel[1,](input, output, index_n, len(index_n), I_N, M, m, n, stride_m, stride_n)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 1, "Expected exactly one call to tts.make_gather_scatter_tptr"

    expected_output = torch.full((m, n), -1, device=device, dtype=torch.float32)
    mask_n = index_n < n
    
    v = input[:,:len(index_n)]
    v = v[:,mask_n]
    index_n = index_n[mask_n]
    expected_output[:,index_n] = v

    torch.testing.assert_close(output, expected_output)


@triton.jit
def unstructured_mask_3d_non_continuous_load_kernel(in_ptr, out_ptr, index_m_ptr, im, I_M:tl.constexpr, N:tl.constexpr, B:tl.constexpr, b, m, n, stride_b, stride_m, stride_n, o_stride_b, o_stride_m, o_stride_n):
    offs_m = tl.arange(0, I_M)
    offs_n = tl.arange(0, N)
    offs_b = tl.arange(0, B)

    index_m = tl.load(index_m_ptr + offs_m, mask=offs_m < im, other=0)
    mask_m_i = index_m < m and offs_m < im

    # dim 1 with unstructured offset.
    in_ptrs = in_ptr + offs_b[:, None, None] * stride_b + index_m[None, :, None] * stride_m + offs_n[None, None, :] * stride_n
    # dim 1 with unstructured mask.
    v = tl.load(in_ptrs, mask=offs_b[:, None, None] < b and mask_m_i[None, :, None] and offs_n[None, None, :] < n, other=-2)

    out_ptrs = out_ptr + offs_b[:, None, None] * o_stride_b + offs_m[None, :, None] * o_stride_m + offs_n[None, None, :] * o_stride_n
    tl.store(out_ptrs, v, mask=offs_b[:, None, None] < b and offs_m[None, :, None] < im and offs_n[None, None, :] < n)

def test_unstructured_mask_3d_non_continuous_load(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    b = 4
    m = 6
    n = 16
    B = triton.next_power_of_2(b)
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (b * m * n), device=device, dtype=torch.float32).reshape(b, m, n)
    index_m = torch.tensor([1, 3, 7], device=device, dtype=torch.int32)

    index_n = torch.tensor([10, 20, 15], device=device, dtype=torch.int32)

    stride_b = input.stride(0)
    stride_m = input.stride(1)
    stride_n = input.stride(2)

    I_M = triton.next_power_of_2(len(index_m))
    I_N = triton.next_power_of_2(len(index_n))

    output = torch.full((b, len(index_m), n), -1, device=device, dtype=torch.float32)

    o_stride_b = output.stride(0)
    o_stride_m = output.stride(1)
    o_stride_n = output.stride(2)

    res = unstructured_mask_3d_non_continuous_load_kernel[1,](input, output, index_m, len(index_m), I_M, N, B, b, m, n, 
                                                         stride_b, stride_m, stride_n,
                                                         o_stride_b, o_stride_m, o_stride_n)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 1, "Expected exactly one call to tts.make_gather_scatter_tptr"

    expected_output = torch.full((b, len(index_m), n), -2, device=device, dtype=torch.float32)
    mask_m = index_m < m
    index_m = index_m[mask_m]
    expected_output[:,:len(index_m),:] = input[:,index_m,:]

    print(output)
    print(expected_output)
    torch.testing.assert_close(output, expected_output)


@triton.jit
def unstructured_mask_3d_non_continuous_store_kernel(in_ptr, out_ptr, index_n_ptr, i_n, I_N:tl.constexpr, M:tl.constexpr, B:tl.constexpr, b, m, n, stride_b, stride_m, stride_n):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, I_N)
    offs_b = tl.arange(0, B)

    index_n = tl.load(index_n_ptr + offs_n, mask=offs_n < n, other=0)
    mask_n_i = index_n < n and offs_n < i_n


    in_ptrs = in_ptr + offs_b[:, None, None] * stride_b + offs_m[None, :, None] * stride_m + offs_n[None, None, :] * stride_n
    v = tl.load(in_ptrs, mask=offs_b[:, None, None] < b and offs_m[None, :, None] < m and offs_n[None, None, :] < i_n, other=-2)

    # dim 2 with unstructured offset.
    out_ptrs = out_ptr + offs_b[:, None, None] * stride_b + offs_m[None, :, None] * stride_m + index_n[None, None, :] * stride_n
    # dim 2 with unstructured mask.
    tl.store(out_ptrs, v, mask=offs_b[:, None, None] < b and offs_m[None, :, None] < m and mask_n_i[None, None, :])

def test_unstructured_mask_3d_non_continuous_store(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    b = 4
    m = 6
    n = 16
    B = triton.next_power_of_2(b)
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (b * m * n), device=device, dtype=torch.float32).reshape(b, m, n)

    index_n = torch.tensor([10, 20, 15, 1, 3, 5, 7, 0, 2, 4, 6, 8], device=device, dtype=torch.int32)

    stride_b = input.stride(0)
    stride_m = input.stride(1)
    stride_n = input.stride(2)

    I_N = triton.next_power_of_2(len(index_n))

    output = torch.full((b, m, n), -1, device=device, dtype=torch.float32)
    res = unstructured_mask_3d_non_continuous_store_kernel[1,](input, output, index_n, len(index_n), I_N, M, B, b, m, n, stride_b, stride_m, stride_n)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 1, "Expected exactly one call to tts.make_gather_scatter_tptr"

    expected_output = torch.full((b, m, n), -1, device=device, dtype=torch.float32)
    mask_n = index_n < n
    
    v = input[:,:,:len(index_n)]
    v = v[:,:,mask_n]
    index_n = index_n[mask_n]
    expected_output[:,:,index_n] = v

    torch.testing.assert_close(output, expected_output)


@triton.jit
def unstructured_mask_and_structured_mask_kernel(in_ptr, out_ptr, m, n, stride_pid, stride_m, stride_n, M:tl.constexpr, N:tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    mask_m = offs_m > pid
    mask_n = offs_n > pid

    in_ptrs = in_ptr + pid * stride_pid + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    # dim 0 with unstructured mask.
    v = tl.load(in_ptrs, mask=offs_m[:, None] < m and offs_n[None, :] < n and mask_m[:, None],
                other=-2)
    out_ptrs = out_ptr + pid * stride_pid + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    # dim 1 with unstructured mask.
    tl.store(out_ptrs, v, mask=offs_m[:, None] < m and offs_n[None, :] < n and mask_n[None, :])


def unstructured_mask_and_structured_mask_torch(input, n_pids, m, n, M, N):
    output = torch.full_like(input, -1)
    tmp = torch.full_like(input, -2)
    for pid in range(n_pids):
        offs_m = torch.arange(0, m)
        offs_n = torch.arange(0, n)

        mask_m = (offs_m > pid)
        mask_n = (offs_n > pid)

        tmp[pid, mask_m, :] = input[pid, mask_m, :]
        output[pid, :, mask_n] = tmp[pid, :, mask_n]

    return output

def test_unstructured_mask_and_structured_mask(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())
    n_pids = 3
    m = 6
    n = 16
    M = triton.next_power_of_2(m)
    N = triton.next_power_of_2(n)
    input = torch.arange(2, 2 + (n_pids * m * n), device=device, dtype=torch.float32).reshape(n_pids, m, n)
    output = torch.full_like(input, -1)

    stride_pid = input.stride(0)
    stride_m = input.stride(1)
    stride_n = input.stride(2)

    res = unstructured_mask_and_structured_mask_kernel[n_pids,](input, output, m, n, stride_pid, stride_m, stride_n, M, N)
    tts_code = run_triton_to_structured(res.asm['ttir'])
    assert tts_code.count('tts.make_gather_scatter_tptr') == 2, "Expected exactly two calls to tts.make_gather_scatter_tptr"

    expected_output = unstructured_mask_and_structured_mask_torch(input, n_pids, m, n, M, N)

    torch.testing.assert_close(output, expected_output)
