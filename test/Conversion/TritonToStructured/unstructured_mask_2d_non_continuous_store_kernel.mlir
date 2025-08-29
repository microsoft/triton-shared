// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with unsturctured mask generate correctly from column-structured ptr with unstructured mask.

// CHECK-LABEL:   tt.func public @generic_mask_2d_non_continuous_store_kernel(
// CHECK-SAME:                                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_7:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_12:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_13:.*]] = tt.splat %[[VAL_5]] : i32 -> tensor<16xi32>
// CHECK:           %[[VAL_14:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_16:.*]] = arith.minsi %[[VAL_15]], %[[VAL_11]] : index
// CHECK:           %[[VAL_17:.*]] = arith.maxsi %[[VAL_16]], %[[VAL_10]] : index
// CHECK:           %[[VAL_18:.*]] = "tts.load"(%[[VAL_14]], %[[VAL_17]], %[[VAL_9]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<16x!tt.ptr<i32>>, index, i32) -> tensor<16xi32>
// CHECK:           %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_13]] : tensor<16xi32>
// CHECK:           %[[VAL_20:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<16xi32>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_20]] : tensor<16xi32>
// CHECK:           %[[VAL_22:.*]] = arith.andi %[[VAL_19]], %[[VAL_21]] : tensor<16xi1>
// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_24:.*]] = tts.make_tptr %[[VAL_0]] to sizes: [8, 16], strides: {{\[}}%[[VAL_23]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<8x16x!tt.ptr<f32>>
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_26:.*]] = arith.minsi %[[VAL_25]], %[[VAL_8]] : index
// CHECK:           %[[VAL_27:.*]] = arith.maxsi %[[VAL_26]], %[[VAL_10]] : index
// CHECK:           %[[VAL_28:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_29:.*]] = arith.minsi %[[VAL_28]], %[[VAL_11]] : index
// CHECK:           %[[VAL_30:.*]] = arith.maxsi %[[VAL_29]], %[[VAL_10]] : index
// CHECK:           %[[VAL_31:.*]] = arith.minsi %[[VAL_27]], %[[VAL_8]] : index
// CHECK:           %[[VAL_32:.*]] = arith.minsi %[[VAL_30]], %[[VAL_11]] : index
// CHECK:           %[[VAL_33:.*]] = "tts.load"(%[[VAL_24]], %[[VAL_31]], %[[VAL_32]], %[[VAL_7]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<8x16x!tt.ptr<f32>>, index, index, f32) -> tensor<8x16xf32>
// CHECK:           %[[VAL_34:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [8, 16] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_18]] gather_scatter_mask: %[[VAL_22]], strides: {{\[}}%[[VAL_23]], 1], offsets: [0, 0] : tensor<16xi32> tensor<16xi1> <f32> to !tt.ptr<tensor<8x16xf32>>
// CHECK:           "tts.store"(%[[VAL_34]], %[[VAL_33]], %[[VAL_31]]) <{static_mask_dims = array<i64: -9223372036854775808, 0>}> : (!tt.ptr<tensor<8x16xf32>>, tensor<8x16xf32>, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @generic_mask_2d_non_continuous_store_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<8x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<16xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.splat %arg5 : i32 -> tensor<16xi32>
    %3 = arith.cmpi slt, %1, %2 : tensor<16xi32>
    %4 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %5 = tt.addptr %4, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %6 = tt.load %5, %3, %cst_0 : tensor<16x!tt.ptr<i32>>
    %7 = arith.cmpi slt, %6, %2 : tensor<16xi32>
    %8 = tt.splat %arg3 : i32 -> tensor<16xi32>
    %9 = arith.cmpi slt, %1, %8 : tensor<16xi32>
    %10 = arith.andi %7, %9 : tensor<16xi1>
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %12 = tt.splat %arg6 : i32 -> tensor<8x1xi32>
    %13 = arith.muli %11, %12 : tensor<8x1xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %16 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %17 = tt.broadcast %15 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
    %18 = tt.broadcast %16 : tensor<1x16xi32> -> tensor<8x16xi32>
    %19 = tt.addptr %17, %18 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %20 = tt.splat %arg4 : i32 -> tensor<8x1xi32>
    %21 = arith.cmpi slt, %11, %20 : tensor<8x1xi32>
    %22 = tt.splat %arg3 : i32 -> tensor<1x16xi32>
    %23 = arith.cmpi slt, %16, %22 : tensor<1x16xi32>
    %24 = tt.broadcast %21 : tensor<8x1xi1> -> tensor<8x16xi1>
    %25 = tt.broadcast %23 : tensor<1x16xi1> -> tensor<8x16xi1>
    %26 = arith.andi %24, %25 : tensor<8x16xi1>
    %27 = tt.load %19, %26, %cst : tensor<8x16x!tt.ptr<f32>>
    %28 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %29 = tt.addptr %28, %13 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %30 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %31 = tt.broadcast %29 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
    %32 = tt.broadcast %30 : tensor<1x16xi32> -> tensor<8x16xi32>
    %33 = tt.addptr %31, %32 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %34 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %35 = tt.broadcast %34 : tensor<1x16xi1> -> tensor<8x16xi1>
    %36 = arith.andi %24, %35 : tensor<8x16xi1>
    tt.store %33, %27, %36 : tensor<8x16x!tt.ptr<f32>>
    tt.return
  }
}