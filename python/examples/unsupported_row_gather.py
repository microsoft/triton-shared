import torch

import triton
import triton.language as tl

def test_unsupported_loop():
    @triton.jit
    def gather_row(
        in0, out0, stride0, stride1
    ):
        offs_2d_x = tl.arange(0, 4) // 4
        offs_2d_y = tl.arange(0, 4)

        offs = offs_2d_x[:, None] * stride0 + offs_2d_y[None, :] * stride1
        in_ptrs = in0 + offs
        out_ptrs = out0 + offs
        for i in range(2):
            a = tl.load(in_ptrs)
            tl.store(out_ptrs, a)
            in_ptrs += 10
            out_ptrs += 10

    src = triton.compiler.ASTSource(
        fn=gather_row,
        signature={"in0": "*fp32", "out0": "*fp32", "stride0": "i32", "stride1": "i32"},
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    # Problem:
    # To make row-gather work in loops, we have to split the loop-iter arg of the tensor of pointer into separate tensors for each dimension.
    # Each iteration will then increment the appropriate tensor.

    #  tt.func public @gather_row(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    #    %c1_i32 = arith.constant 1 : i32
    #    %c2_i32 = arith.constant 2 : i32
    #    %c0_i32 = arith.constant 0 : i32
    #    %cst = arith.constant dense<10> : tensor<4x4xi32>
    #    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    #    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    #    %1 = arith.divsi %0, %cst_0 : tensor<4xi32>
    #    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    #    %3 = tt.splat %arg2 : i32 -> tensor<4x1xi32>
    #    %4 = arith.muli %2, %3 : tensor<4x1xi32>
    #    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    #    %6 = tt.splat %arg3 : i32 -> tensor<1x4xi32>
    #    %7 = arith.muli %5, %6 : tensor<1x4xi32>
    #    %8 = tt.broadcast %4 : tensor<4x1xi32> -> tensor<4x4xi32>
    #    %9 = tt.broadcast %7 : tensor<1x4xi32> -> tensor<4x4xi32>
    #    %10 = arith.addi %8, %9 : tensor<4x4xi32>
    #    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    #    %12 = tt.addptr %11, %10 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    #    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    #    %14 = tt.addptr %13, %10 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    #    %15:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %12, %arg6 = %14) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
    #      %16 = tt.load %arg5 : tensor<4x4x!tt.ptr<f32>>
    #      tt.store %arg6, %16 : tensor<4x4x!tt.ptr<f32>>
    #      %17 = tt.addptr %arg5, %cst : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    #      %18 = tt.addptr %arg6, %cst : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    #      scf.yield %17, %18 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    #    }
    #    tt.return
    #  }
