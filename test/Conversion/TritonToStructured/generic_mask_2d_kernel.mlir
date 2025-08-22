// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with generic mask generate correctly.

// CHECK-LABEL:   tt.func public @generic_mask_2d_kernel(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i8> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i8> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_6:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant dense<0> : tensor<16xi32>
// CHECK:           %[[VAL_12:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK:           %[[VAL_13:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i8> to tensor<8x!tt.ptr<i8>>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_15:.*]] = arith.minsi %[[VAL_14]], %[[VAL_10]] : index
// CHECK:           %[[VAL_16:.*]] = arith.maxsi %[[VAL_15]], %[[VAL_9]] : index
// CHECK:           %[[VAL_17:.*]] = "tts.load"(%[[VAL_13]], %[[VAL_16]], %[[VAL_8]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<8x!tt.ptr<i8>>, index, i8) -> tensor<8xi8>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : tensor<8xi8> to tensor<8xi32>
// CHECK:           %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_12]] : tensor<8xi32>
// CHECK:           %[[VAL_20:.*]] = tts.make_tptr %[[VAL_3]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <i8> to tensor<16x!tt.ptr<i8>>
// CHECK:           %[[VAL_21:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_22:.*]] = arith.minsi %[[VAL_21]], %[[VAL_7]] : index
// CHECK:           %[[VAL_23:.*]] = arith.maxsi %[[VAL_22]], %[[VAL_9]] : index
// CHECK:           %[[VAL_24:.*]] = "tts.load"(%[[VAL_20]], %[[VAL_23]], %[[VAL_8]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<16x!tt.ptr<i8>>, index, i8) -> tensor<16xi8>
// CHECK:           %[[VAL_25:.*]] = arith.extsi %[[VAL_24]] : tensor<16xi8> to tensor<16xi32>
// CHECK:           %[[VAL_26:.*]] = arith.cmpi ne, %[[VAL_25]], %[[VAL_11]] : tensor<16xi32>
// CHECK:           %[[VAL_27:.*]] = arith.minsi %[[VAL_23]], %[[VAL_7]] : index
// CHECK:           %[[VAL_28:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_29:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [8, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_28]] gather_scatter_mask: %[[VAL_19]], strides: {{\[}}%[[VAL_7]], 1], offsets: [0, 0] : tensor<8xi32> tensor<8xi1> <f32> to !tt.ptr<tensor<8x16xf32>>
// CHECK:           %[[VAL_30:.*]] = "tts.load"(%[[VAL_29]], %[[VAL_27]], %[[VAL_6]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: 0, -9223372036854775808>}> : (!tt.ptr<tensor<8x16xf32>>, index, f32) -> tensor<8x16xf32>
// CHECK:           %[[VAL_31:.*]] = arith.minsi %[[VAL_16]], %[[VAL_10]] : index
// CHECK:           %[[VAL_32:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_33:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [8, 16] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_32]] gather_scatter_mask: %[[VAL_26]], strides: {{\[}}%[[VAL_7]], 1], offsets: [0, 0] : tensor<16xi32> tensor<16xi1> <f32> to !tt.ptr<tensor<8x16xf32>>
// CHECK:           "tts.store"(%[[VAL_33]], %[[VAL_30]], %[[VAL_31]]) <{static_mask_dims = array<i64: -9223372036854775808, 0>}> : (!tt.ptr<tensor<8x16xf32>>, tensor<8x16xf32>, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @generic_mask_2d_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<8x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<16xi8>
    %cst_1 = arith.constant dense<0> : tensor<8xi8>
    %cst_2 = arith.constant dense<16> : tensor<8x1xi32>
    %cst_3 = arith.constant dense<0> : tensor<16xi32>
    %cst_4 = arith.constant dense<0> : tensor<8xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.splat %arg4 : i32 -> tensor<8xi32>
    %3 = arith.cmpi slt, %0, %2 : tensor<8xi32>
    %4 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %5 = tt.addptr %4, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %6 = tt.load %5, %3, %cst_1 : tensor<8x!tt.ptr<i8>>
    %7 = arith.extsi %6 : tensor<8xi8> to tensor<8xi32>
    %8 = arith.cmpi ne, %7, %cst_4 : tensor<8xi32>
    %9 = tt.splat %arg5 : i32 -> tensor<16xi32>
    %10 = arith.cmpi slt, %1, %9 : tensor<16xi32>
    %11 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %12 = tt.addptr %11, %1 : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %13 = tt.load %12, %10, %cst_0 : tensor<16x!tt.ptr<i8>>
    %14 = arith.extsi %13 : tensor<16xi8> to tensor<16xi32>
    %15 = arith.cmpi ne, %14, %cst_3 : tensor<16xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %17 = arith.muli %16, %cst_2 : tensor<8x1xi32>
    %18 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %17 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %20 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %21 = tt.broadcast %19 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
    %22 = tt.broadcast %20 : tensor<1x16xi32> -> tensor<8x16xi32>
    %23 = tt.addptr %21, %22 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %24 = tt.expand_dims %8 {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
    %25 = tt.splat %arg5 : i32 -> tensor<1x16xi32>
    %26 = arith.cmpi slt, %20, %25 : tensor<1x16xi32>
    %27 = tt.broadcast %24 : tensor<8x1xi1> -> tensor<8x16xi1>
    %28 = tt.broadcast %26 : tensor<1x16xi1> -> tensor<8x16xi1>
    %29 = arith.andi %27, %28 : tensor<8x16xi1>
    %30 = tt.load %23, %29, %cst : tensor<8x16x!tt.ptr<f32>>
    %31 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %32 = tt.addptr %31, %17 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %33 = tt.broadcast %32 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
    %34 = tt.addptr %33, %22 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %35 = tt.splat %arg4 : i32 -> tensor<8x1xi32>
    %36 = arith.cmpi slt, %16, %35 : tensor<8x1xi32>
    %37 = tt.expand_dims %15 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %38 = tt.broadcast %36 : tensor<8x1xi1> -> tensor<8x16xi1>
    %39 = tt.broadcast %37 : tensor<1x16xi1> -> tensor<8x16xi1>
    %40 = arith.andi %38, %39 : tensor<8x16xi1>
    tt.store %34, %30, %40 : tensor<8x16x!tt.ptr<f32>>
    tt.return
  }
}