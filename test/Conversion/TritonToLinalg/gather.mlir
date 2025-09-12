// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @gather_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<4x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<4x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = tt.broadcast %2 : tensor<4x1xi32> -> tensor<4x4xi32>
    %5 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<4x4xi32>
    %6 = arith.addi %4, %5 : tensor<4x4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %9 = tt.load %8 : tensor<4x4x!tt.ptr<f32>>
    %10 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %12 = arith.muli %11, %cst : tensor<8x1xi32>
    %13 = tt.broadcast %12 : tensor<8x1xi32> -> tensor<8x4xi32>
    %14 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<8x4xi32>
    %15 = arith.addi %13, %14 : tensor<8x4xi32>
    %16 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x4x!tt.ptr<i64>>
    %17 = tt.addptr %16, %15 : tensor<8x4x!tt.ptr<i64>>, tensor<8x4xi32>
    %18 = tt.load %17 : tensor<8x4x!tt.ptr<i64>>
    %19 = tt.gather %9[%18] {axis = 0 : i32} : (tensor<4x4xf32>, tensor<8x4xi64>) -> tensor<8x4xf32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %15 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %21, %19 : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }
}
