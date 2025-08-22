// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with generic mask generate correctly.

// CHECK-LABEL:   tt.func public @generic_mask_3d_kernel(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i8> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i8> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_9:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant dense<0> : tensor<16xi32>
// CHECK:           %[[VAL_16:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK:           %[[VAL_17:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i8> to tensor<8x!tt.ptr<i8>>
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = arith.minsi %[[VAL_18]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.maxsi %[[VAL_19]], %[[VAL_13]] : index
// CHECK:           %[[VAL_21:.*]] = "tts.load"(%[[VAL_17]], %[[VAL_20]], %[[VAL_12]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<8x!tt.ptr<i8>>, index, i8) -> tensor<8xi8>
// CHECK:           %[[VAL_22:.*]] = arith.extsi %[[VAL_21]] : tensor<8xi8> to tensor<8xi32>
// CHECK:           %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_16]] : tensor<8xi32>
// CHECK:           %[[VAL_24:.*]] = tts.make_tptr %[[VAL_3]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <i8> to tensor<16x!tt.ptr<i8>>
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_26:.*]] = arith.minsi %[[VAL_25]], %[[VAL_11]] : index
// CHECK:           %[[VAL_27:.*]] = arith.maxsi %[[VAL_26]], %[[VAL_13]] : index
// CHECK:           %[[VAL_28:.*]] = "tts.load"(%[[VAL_24]], %[[VAL_27]], %[[VAL_12]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<16x!tt.ptr<i8>>, index, i8) -> tensor<16xi8>
// CHECK:           %[[VAL_29:.*]] = arith.extsi %[[VAL_28]] : tensor<16xi8> to tensor<16xi32>
// CHECK:           %[[VAL_30:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_15]] : tensor<16xi32>
// CHECK:           %[[VAL_31:.*]] = arith.index_cast %[[VAL_7]] : i32 to index
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_34:.*]] = arith.minsi %[[VAL_33]], %[[VAL_10]] : index
// CHECK:           %[[VAL_35:.*]] = arith.maxsi %[[VAL_34]], %[[VAL_13]] : index
// CHECK:           %[[VAL_36:.*]] = arith.minsi %[[VAL_27]], %[[VAL_11]] : index
// CHECK:           %[[VAL_37:.*]] = arith.minsi %[[VAL_35]], %[[VAL_10]] : index
// CHECK:           %[[VAL_38:.*]] = arith.minsi %[[VAL_36]], %[[VAL_11]] : index
// CHECK:           %[[VAL_39:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_40:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 8, 16] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_39]] gather_scatter_mask: %[[VAL_23]], strides: {{\[}}%[[VAL_31]], %[[VAL_32]], 1], offsets: [0, 0, 0] : tensor<8xi32> tensor<8xi1> <f32> to !tt.ptr<tensor<4x8x16xf32>>
// CHECK:           %[[VAL_41:.*]] = "tts.load"(%[[VAL_40]], %[[VAL_37]], %[[VAL_38]], %[[VAL_9]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, 0, -9223372036854775808>}> : (!tt.ptr<tensor<4x8x16xf32>>, index, index, f32) -> tensor<4x8x16xf32>
// CHECK:           %[[VAL_42:.*]] = arith.minsi %[[VAL_20]], %[[VAL_14]] : index
// CHECK:           %[[VAL_43:.*]] = arith.minsi %[[VAL_42]], %[[VAL_14]] : index
// CHECK:           %[[VAL_44:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_45:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4, 8, 16] gather_scatter_dim: 2 gather_scatter_offset: %[[VAL_44]] gather_scatter_mask: %[[VAL_30]], strides: {{\[}}%[[VAL_31]], %[[VAL_32]], 1], offsets: [0, 0, 0] : tensor<16xi32> tensor<16xi1> <f32> to !tt.ptr<tensor<4x8x16xf32>>
// CHECK:           "tts.store"(%[[VAL_45]], %[[VAL_41]], %[[VAL_37]], %[[VAL_43]]) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808, 0>}> : (!tt.ptr<tensor<4x8x16xf32>>, tensor<4x8x16xf32>, index, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @generic_mask_3d_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<4x8x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<16xi8>
    %cst_1 = arith.constant dense<0> : tensor<8xi8>
    %cst_2 = arith.constant dense<0> : tensor<16xi32>
    %cst_3 = arith.constant dense<0> : tensor<8xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %arg5 : i32 -> tensor<8xi32>
    %4 = arith.cmpi slt, %0, %3 : tensor<8xi32>
    %5 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %7 = tt.load %6, %4, %cst_1 : tensor<8x!tt.ptr<i8>>
    %8 = arith.extsi %7 : tensor<8xi8> to tensor<8xi32>
    %9 = arith.cmpi ne, %8, %cst_3 : tensor<8xi32>
    %10 = tt.splat %arg6 : i32 -> tensor<16xi32>
    %11 = arith.cmpi slt, %1, %10 : tensor<16xi32>
    %12 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %13 = tt.addptr %12, %1 : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %14 = tt.load %13, %11, %cst_0 : tensor<16x!tt.ptr<i8>>
    %15 = arith.extsi %14 : tensor<16xi8> to tensor<16xi32>
    %16 = arith.cmpi ne, %15, %cst_2 : tensor<16xi32>
    %17 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %18 = tt.expand_dims %17 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %19 = tt.splat %arg7 : i32 -> tensor<4x1x1xi32>
    %20 = arith.muli %18, %19 : tensor<4x1x1xi32>
    %21 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %22 = tt.addptr %21, %20 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %23 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %24 = tt.expand_dims %23 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %25 = tt.splat %arg8 : i32 -> tensor<1x8x1xi32>
    %26 = arith.muli %24, %25 : tensor<1x8x1xi32>
    %27 = tt.broadcast %22 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x8x1x!tt.ptr<f32>>
    %28 = tt.broadcast %26 : tensor<1x8x1xi32> -> tensor<4x8x1xi32>
    %29 = tt.addptr %27, %28 : tensor<4x8x1x!tt.ptr<f32>>, tensor<4x8x1xi32>
    %30 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %32 = tt.broadcast %29 : tensor<4x8x1x!tt.ptr<f32>> -> tensor<4x8x16x!tt.ptr<f32>>
    %33 = tt.broadcast %31 : tensor<1x1x16xi32> -> tensor<4x8x16xi32>
    %34 = tt.addptr %32, %33 : tensor<4x8x16x!tt.ptr<f32>>, tensor<4x8x16xi32>
    %35 = tt.splat %arg4 : i32 -> tensor<4x1x1xi32>
    %36 = arith.cmpi slt, %18, %35 : tensor<4x1x1xi32>
    %37 = tt.expand_dims %9 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
    %38 = tt.expand_dims %37 {axis = 2 : i32} : tensor<1x8xi1> -> tensor<1x8x1xi1>
    %39 = tt.splat %arg6 : i32 -> tensor<1x1x16xi32>
    %40 = arith.cmpi slt, %31, %39 : tensor<1x1x16xi32>
    %41 = tt.broadcast %38 : tensor<1x8x1xi1> -> tensor<1x8x16xi1>
    %42 = tt.broadcast %40 : tensor<1x1x16xi1> -> tensor<1x8x16xi1>
    %43 = arith.andi %41, %42 : tensor<1x8x16xi1>
    %44 = tt.broadcast %36 : tensor<4x1x1xi1> -> tensor<4x8x16xi1>
    %45 = tt.broadcast %43 : tensor<1x8x16xi1> -> tensor<4x8x16xi1>
    %46 = arith.andi %44, %45 : tensor<4x8x16xi1>
    %47 = tt.load %34, %46, %cst : tensor<4x8x16x!tt.ptr<f32>>
    %48 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %49 = tt.addptr %48, %20 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %50 = tt.broadcast %49 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x8x1x!tt.ptr<f32>>
    %51 = tt.addptr %50, %28 : tensor<4x8x1x!tt.ptr<f32>>, tensor<4x8x1xi32>
    %52 = tt.broadcast %51 : tensor<4x8x1x!tt.ptr<f32>> -> tensor<4x8x16x!tt.ptr<f32>>
    %53 = tt.addptr %52, %33 : tensor<4x8x16x!tt.ptr<f32>>, tensor<4x8x16xi32>
    %54 = tt.splat %arg5 : i32 -> tensor<1x8x1xi32>
    %55 = arith.cmpi slt, %24, %54 : tensor<1x8x1xi32>
    %56 = tt.expand_dims %16 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %57 = tt.expand_dims %56 {axis = 1 : i32} : tensor<1x16xi1> -> tensor<1x1x16xi1>
    %58 = tt.broadcast %55 : tensor<1x8x1xi1> -> tensor<1x8x16xi1>
    %59 = tt.broadcast %57 : tensor<1x1x16xi1> -> tensor<1x8x16xi1>
    %60 = arith.andi %58, %59 : tensor<1x8x16xi1>
    %61 = tt.broadcast %60 : tensor<1x8x16xi1> -> tensor<4x8x16xi1>
    %62 = arith.andi %44, %61 : tensor<4x8x16xi1>
    tt.store %53, %47, %62 : tensor<4x8x16x!tt.ptr<f32>>
    tt.return
  }
}