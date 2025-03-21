// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s

// CHECK: %[[CST:.*]] = arith.constant dense<8> : tensor<8xi32>
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[CST_0:.*]] = arith.constant dense<3> : tensor<8xi32>
// CHECK: %[[CST_1:.*]] = arith.constant dense<5> : tensor<8xi32>
// CHECK: %[[TPTR:.*]] = tts.make_tptr %arg1 to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<8x!tt.ptr<i32>>
// CHECK: %[[LOAD1:.*]] = "tts.load"(%[[TPTR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<i32>>) -> tensor<8xi32>
// CHECK: %[[MULI1:.*]] = arith.muli %[[LOAD1]], %[[CST_0]] : tensor<8xi32>
// CHECK: %[[REMSI:.*]] = arith.remsi %[[MULI1]], %[[CST_1]] : tensor<8xi32>
// CHECK: %[[MULI2:.*]] = arith.muli %[[REMSI]], %[[CST]] : tensor<8xi32>
// CHECK: %[[INDIRECT_TPTR:.*]] = tts.make_indirect_tptr %arg0 to sizes: [8, 8] indirect_dim: 0 indirect_offset: %[[MULI2]], strides: [1, 1], offsets: [0, 0] : tensor<8xi32> <f32> to !tt.ptr<tensor<8x8xf32>>
// CHECK: "tts.load"(%[[INDIRECT_TPTR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x8xf32>>) -> tensor<8x8xf32>

module attributes {maia.triton_kernel} {
  tt.func public @row_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<8x1xi32>
    %cst_0 = arith.constant dense<8> : tensor<8xi32>
    %cst_1 = arith.constant dense<3> : tensor<8xi32>
    %cst_2 = arith.constant dense<5> : tensor<8xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = arith.muli %3, %cst_1 : tensor<8xi32>
    %5 = arith.remsi %4, %cst_2 : tensor<8xi32>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %7 = arith.muli %6, %cst : tensor<8x1xi32>
    %8 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %9 = tt.broadcast %7 : tensor<8x1xi32> -> tensor<8x8xi32>
    %10 = tt.broadcast %8 : tensor<1x8xi32> -> tensor<8x8xi32>
    %11 = arith.addi %9, %10 : tensor<8x8xi32>
    %12 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %13 = tt.addptr %12, %11 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %14 = tt.load %13 : tensor<8x8x!tt.ptr<f32>>
    %15 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %16 = arith.muli %15, %cst : tensor<8x1xi32>
    %17 = tt.broadcast %16 : tensor<8x1xi32> -> tensor<8x8xi32>
    %18 = arith.addi %17, %10 : tensor<8x8xi32>
    %19 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    tt.store %20, %14 : tensor<8x8x!tt.ptr<f32>>
    tt.return
  }
}
