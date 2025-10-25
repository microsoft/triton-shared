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

    # Problems:
    # 1) Row-gather sub-problems
    #  At a high level, the current row-gather implementation reuses the PtrAnalysis
    #  visitor functions to detect when a dimension is not structured. If a visitor
    #  function fails, we assume that the original triton SSA value that causes the
    #  the failure is the tensor representing the offset for that unstructured dim.
    #  (need to verify if this is indeed always true).
    #  Some problems with this approach:
    #  - relies on existing PtrAnalysis visitor functions, which handle certain op
    #    differently (e.g: remsi)
    #  - make the PtrAnalysis code more complicated since it now handles both
    #    structured and unstructured case
    # I think another way to break this problem down is to:
    # a) have a separate "dimension" analysis that, given tensor of pointer
    # expression, knows how to generate the tensor of offset for each dimension.
    #
    # b) refactor PtrAnalysis to allow "partially" structured tensors. Right now
    # if a dimension is unstructured, the analysis fails.
    #
    # Combining both a) and b), we can separate out the 2 problems and make the
    # analysis more robust. I have not thought about how both a) and b)
    # will interact with loop support.
    #
    # 2) Tracking stride
    # Tracking stride in the gather (unstructured) dimension is complicated
    # (most of the code is in PtrState::addState). The main problem is given
    # a tensor of pointer expression, for each dimension, what is the stride?
    # The current code is examining many cases which may be error-prone. How do
    # we solve this in a more generalized way?
    #
    # 3) Loop
    # To make row-gather work in loops, we have to split the loop-iter arg of the
    # tensor of pointers into separate tensors for each dimension.
    # Each iteration will then increment the appropriate tensor.
    # The current row-gather approach may work with some changes (need further
    # investigation), but since we already have plans to refactor this code,
    # it may be easier to implement a new approach from scratch to decouple
    # the row-gather from the original PtrAnalysis implementation.
    #
    # 4) Mask
    # The current mask analysis processes the full mask, but for row-gather, we
    # want to only compute mask for the structured dimension only. How do we split
    # the combined mask value to get the structured and unstructured dimensions?
    # For example, for a 2D boolean tensor, we want to split it into 2 1D tensors,
    # one for the gather (unstructured) dimension, the other for the structured
    # dimension.

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
