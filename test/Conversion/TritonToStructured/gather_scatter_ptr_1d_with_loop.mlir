// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// TODO: fix test case in https://github.com/microsoft/triton-shared/pull/332, remove XFAIL and update the CHECKs.
// XFAIL: *

// Make sure tts.make_gather_scatter_tptr is generated with for 1D tensor on addptr with loop.

// CHECK: make_gather_scatter_tptr

module {
  tt.func public @gather_row(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<10> : tensor<4xi32>
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<4xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %3, %arg6 = %5) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>)  : i32 {
      %7 = tt.load %arg5 : tensor<4x!tt.ptr<f32>>
      tt.store %arg6, %7 : tensor<4x!tt.ptr<f32>>
      %8 = tt.addptr %arg5, %cst : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %9 = tt.addptr %arg6, %cst : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %8, %9 : tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}
