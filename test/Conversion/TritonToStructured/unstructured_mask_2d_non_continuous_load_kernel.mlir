// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with unsturctured mask generate correctly from row-structured ptr with unstructured mask.

// CHECK-LABEL:   tt.func public @generic_mask_2d_non_continuous_load_kernel(
// CHECK-SAME:                                                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                               %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                               %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                               %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                               %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_7:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_12:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_13:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<4xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_13]] : tensor<4xi32>
// CHECK:           %[[VAL_15:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
// CHECK:           %[[VAL_16:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_17:.*]] = arith.minsi %[[VAL_16]], %[[VAL_11]] : index
// CHECK:           %[[VAL_18:.*]] = arith.maxsi %[[VAL_17]], %[[VAL_10]] : index
// CHECK:           %[[VAL_19:.*]] = "tts.load"(%[[VAL_15]], %[[VAL_18]], %[[VAL_9]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<i32>>, index, i32) -> tensor<4xi32>
// CHECK:           %[[VAL_20:.*]] = tt.splat %[[VAL_4]] : i32 -> tensor<4xi32>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_20]] : tensor<4xi32>
// CHECK:           %[[VAL_22:.*]] = arith.andi %[[VAL_21]], %[[VAL_14]] : tensor<4xi1>
// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_24:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_25:.*]] = arith.minsi %[[VAL_24]], %[[VAL_8]] : index
// CHECK:           %[[VAL_26:.*]] = arith.maxsi %[[VAL_25]], %[[VAL_10]] : index
// CHECK:           %[[VAL_27:.*]] = arith.minsi %[[VAL_26]], %[[VAL_8]] : index
// CHECK:           %[[VAL_28:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_19]] gather_scatter_mask: %[[VAL_22]], strides: {{\[}}%[[VAL_23]], 1], offsets: [0, 0] : tensor<4xi32> tensor<4xi1> <f32> to !tt.ptr<tensor<4x16xf32>>
// CHECK:           %[[VAL_29:.*]] = "tts.load"(%[[VAL_28]], %[[VAL_27]], %[[VAL_7]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: 0, -9223372036854775808>}> : (!tt.ptr<tensor<4x16xf32>>, index, f32) -> tensor<4x16xf32>


module {
  tt.func public @generic_mask_2d_non_continuous_load_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<4x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %3 = arith.cmpi slt, %0, %2 : tensor<4xi32>
    %4 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %6 = tt.load %5, %3, %cst_0 : tensor<4x!tt.ptr<i32>>
    %7 = tt.splat %arg4 : i32 -> tensor<4xi32>
    %8 = arith.cmpi slt, %6, %7 : tensor<4xi32>
    %9 = arith.andi %8, %3 : tensor<4xi1>
    %10 = tt.expand_dims %6 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %11 = tt.splat %arg6 : i32 -> tensor<4x1xi32>
    %12 = arith.muli %10, %11 : tensor<4x1xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %14 = tt.addptr %13, %12 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %15 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %16 = tt.broadcast %14 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %17 = tt.broadcast %15 : tensor<1x16xi32> -> tensor<4x16xi32>
    %18 = tt.addptr %16, %17 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %19 = tt.expand_dims %9 {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %20 = tt.splat %arg5 : i32 -> tensor<1x16xi32>
    %21 = arith.cmpi slt, %15, %20 : tensor<1x16xi32>
    %22 = tt.broadcast %19 : tensor<4x1xi1> -> tensor<4x16xi1>
    %23 = tt.broadcast %21 : tensor<1x16xi1> -> tensor<4x16xi1>
    %24 = arith.andi %22, %23 : tensor<4x16xi1>
    %25 = tt.load %18, %24, %cst : tensor<4x16x!tt.ptr<f32>>
    %26 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %27 = arith.muli %26, %11 : tensor<4x1xi32>
    %28 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %29 = tt.addptr %28, %27 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %30 = tt.broadcast %29 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
    %31 = tt.addptr %30, %17 : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %32 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %33 = arith.cmpi slt, %26, %32 : tensor<4x1xi32>
    %34 = tt.broadcast %33 : tensor<4x1xi1> -> tensor<4x16xi1>
    %35 = arith.andi %34, %23 : tensor<4x16xi1>
    tt.store %31, %25, %35 : tensor<4x16x!tt.ptr<f32>>
    tt.return
  }
}
