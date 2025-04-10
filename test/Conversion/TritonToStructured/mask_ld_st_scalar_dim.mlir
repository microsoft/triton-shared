// RUN: %triton-opt --triton-to-structured --remove-dead-values --canonicalize %s | %FileCheck %s

module {
  tt.func @mask_ld_st_scalar(
      %arg0: !tt.ptr<f32>,
      %arg1: !tt.ptr<f32>,
      %arg2: i32,
      %arg3: i32,
      %arg4: i32,
      %arg5: i32,
      %arg6: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg2 : i32 -> tensor<2xi32>
    %2 = arith.cmpi slt, %0, %1 : tensor<2xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<2xi32>
    %4 = arith.cmpi slt, %0, %3 : tensor<2xi32>
    %5 = arith.cmpi sgt, %arg4, %c0_i32 : i32
    %6 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %7 = tt.splat %arg6 : i32 -> tensor<2x1xi32>
    %8 = arith.muli %6, %7 : tensor<2x1xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %11 = tt.expand_dims %4 {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
    %12 = tt.splat %5 : i1 -> tensor<2x1xi1>
    %13 = arith.andi %11, %12 : tensor<2x1xi1>
    %14 = tt.load %10, %13 : tensor<2x1x!tt.ptr<f32>>
    %15 = tt.splat %arg5 : i32 -> tensor<2x1xi32>
    %16 = arith.muli %6, %15 : tensor<2x1xi32>
    %17 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %19 = tt.expand_dims %2 {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
    %20 = arith.andi %19, %12 : tensor<2x1xi1>
    %21 = tt.load %18, %20 : tensor<2x1x!tt.ptr<f32>>
    tt.store %10, %14, %13 : tensor<2x1x!tt.ptr<f32>>
    tt.store %18, %21, %20 : tensor<2x1x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:   %{{.*}} = "tts.load"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, index) -> tensor<2x1xf32>
// CHECK:   %{{.*}} = "tts.load"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, index) -> tensor<2x1xf32>
// CHECK:   "tts.store"(%{{.*}}, %{{.*}}, %{{.*}}) <{static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, tensor<2x1xf32>, index) -> ()
// CHECK:   "tts.store"(%{{.*}}, %{{.*}}, %{{.*}}) <{static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, tensor<2x1xf32>, index) -> ()
