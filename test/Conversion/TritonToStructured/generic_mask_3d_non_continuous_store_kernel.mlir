// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with generic mask generate correctly.

// CHECK-LABEL:   tt.func public @generic_mask_3d_non_continuous_store_kernel(
// CHECK-SAME:                                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_9:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_15:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_6]] : i32 -> tensor<16xi32>
// CHECK:           %[[VAL_17:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = arith.minsi %[[VAL_18]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.maxsi %[[VAL_19]], %[[VAL_13]] : index
// CHECK:           %[[VAL_21:.*]] = "tts.load"(%[[VAL_17]], %[[VAL_20]], %[[VAL_12]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<16x!tt.ptr<i32>>, index, i32) -> tensor<16xi32>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_16]] : tensor<16xi32>
// CHECK:           %[[VAL_23:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<16xi32>
// CHECK:           %[[VAL_24:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_23]] : tensor<16xi32>
// CHECK:           %[[VAL_25:.*]] = arith.andi %[[VAL_22]], %[[VAL_24]] : tensor<16xi1>
// CHECK:           %[[VAL_26:.*]] = arith.index_cast %[[VAL_7]] : i32 to index
// CHECK:           %[[VAL_27:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_28:.*]] = tts.make_tptr %[[VAL_0]] to sizes: [4, 8, 16], strides: {{\[}}%[[VAL_26]], %[[VAL_27]], 1], offsets: [0, 0, 0], shape: [0, 0, 0], order: [] : <f32> to tensor<4x8x16x!tt.ptr<f32>>
// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_30:.*]] = arith.minsi %[[VAL_29]], %[[VAL_11]] : index
// CHECK:           %[[VAL_31:.*]] = arith.maxsi %[[VAL_30]], %[[VAL_13]] : index
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_33:.*]] = arith.minsi %[[VAL_32]], %[[VAL_10]] : index
// CHECK:           %[[VAL_34:.*]] = arith.maxsi %[[VAL_33]], %[[VAL_13]] : index
// CHECK:           %[[VAL_35:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_36:.*]] = arith.minsi %[[VAL_35]], %[[VAL_14]] : index
// CHECK:           %[[VAL_37:.*]] = arith.maxsi %[[VAL_36]], %[[VAL_13]] : index
// CHECK:           %[[VAL_38:.*]] = arith.minsi %[[VAL_34]], %[[VAL_10]] : index
// CHECK:           %[[VAL_39:.*]] = arith.minsi %[[VAL_37]], %[[VAL_14]] : index
// CHECK:           %[[VAL_40:.*]] = arith.minsi %[[VAL_31]], %[[VAL_11]] : index
// CHECK:           %[[VAL_41:.*]] = arith.minsi %[[VAL_38]], %[[VAL_10]] : index
// CHECK:           %[[VAL_42:.*]] = arith.minsi %[[VAL_39]], %[[VAL_14]] : index
// CHECK:           %[[VAL_43:.*]] = "tts.load"(%[[VAL_28]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_9]]) <{operandSegmentSizes = array<i32: 1, 3, 1>, static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (tensor<4x8x16x!tt.ptr<f32>>, index, index, index, f32) -> tensor<4x8x16xf32>
// CHECK:           %[[VAL_44:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4, 8, 16] gather_scatter_dim: 2 gather_scatter_offset: %[[VAL_21]] gather_scatter_mask: %[[VAL_25]], strides: {{\[}}%[[VAL_26]], %[[VAL_27]], 1], offsets: [0, 0, 0] : tensor<16xi32> tensor<16xi1> <f32> to !tt.ptr<tensor<4x8x16xf32>>
// CHECK:           "tts.store"(%[[VAL_44]], %[[VAL_43]], %[[VAL_40]], %[[VAL_41]]) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808, 0>}> : (!tt.ptr<tensor<4x8x16xf32>>, tensor<4x8x16xf32>, index, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @generic_mask_3d_non_continuous_store_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<4x8x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<16xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %arg6 : i32 -> tensor<16xi32>
    %4 = arith.cmpi slt, %1, %3 : tensor<16xi32>
    %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.addptr %5, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %7 = tt.load %6, %4, %cst_0 : tensor<16x!tt.ptr<i32>>
    %8 = arith.cmpi slt, %7, %3 : tensor<16xi32>
    %9 = tt.splat %arg3 : i32 -> tensor<16xi32>
    %10 = arith.cmpi slt, %1, %9 : tensor<16xi32>
    %11 = arith.andi %8, %10 : tensor<16xi1>
    %12 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %13 = tt.expand_dims %12 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %14 = tt.splat %arg7 : i32 -> tensor<4x1x1xi32>
    %15 = arith.muli %13, %14 : tensor<4x1x1xi32>
    %16 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %17 = tt.addptr %16, %15 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %18 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %19 = tt.expand_dims %18 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %20 = tt.splat %arg8 : i32 -> tensor<1x8x1xi32>
    %21 = arith.muli %19, %20 : tensor<1x8x1xi32>
    %22 = tt.broadcast %17 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x8x1x!tt.ptr<f32>>
    %23 = tt.broadcast %21 : tensor<1x8x1xi32> -> tensor<4x8x1xi32>
    %24 = tt.addptr %22, %23 : tensor<4x8x1x!tt.ptr<f32>>, tensor<4x8x1xi32>
    %25 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %27 = tt.broadcast %24 : tensor<4x8x1x!tt.ptr<f32>> -> tensor<4x8x16x!tt.ptr<f32>>
    %28 = tt.broadcast %26 : tensor<1x1x16xi32> -> tensor<4x8x16xi32>
    %29 = tt.addptr %27, %28 : tensor<4x8x16x!tt.ptr<f32>>, tensor<4x8x16xi32>
    %30 = tt.splat %arg4 : i32 -> tensor<4x1x1xi32>
    %31 = arith.cmpi slt, %13, %30 : tensor<4x1x1xi32>
    %32 = tt.splat %arg5 : i32 -> tensor<1x8x1xi32>
    %33 = arith.cmpi slt, %19, %32 : tensor<1x8x1xi32>
    %34 = tt.splat %arg3 : i32 -> tensor<1x1x16xi32>
    %35 = arith.cmpi slt, %26, %34 : tensor<1x1x16xi32>
    %36 = tt.broadcast %33 : tensor<1x8x1xi1> -> tensor<1x8x16xi1>
    %37 = tt.broadcast %35 : tensor<1x1x16xi1> -> tensor<1x8x16xi1>
    %38 = arith.andi %36, %37 : tensor<1x8x16xi1>
    %39 = tt.broadcast %31 : tensor<4x1x1xi1> -> tensor<4x8x16xi1>
    %40 = tt.broadcast %38 : tensor<1x8x16xi1> -> tensor<4x8x16xi1>
    %41 = arith.andi %39, %40 : tensor<4x8x16xi1>
    %42 = tt.load %29, %41, %cst : tensor<4x8x16x!tt.ptr<f32>>
    %43 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %44 = tt.addptr %43, %15 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %45 = tt.broadcast %44 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x8x1x!tt.ptr<f32>>
    %46 = tt.addptr %45, %23 : tensor<4x8x1x!tt.ptr<f32>>, tensor<4x8x1xi32>
    %47 = tt.expand_dims %7 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %49 = tt.broadcast %46 : tensor<4x8x1x!tt.ptr<f32>> -> tensor<4x8x16x!tt.ptr<f32>>
    %50 = tt.broadcast %48 : tensor<1x1x16xi32> -> tensor<4x8x16xi32>
    %51 = tt.addptr %49, %50 : tensor<4x8x16x!tt.ptr<f32>>, tensor<4x8x16xi32>
    %52 = tt.expand_dims %11 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %53 = tt.expand_dims %52 {axis = 1 : i32} : tensor<1x16xi1> -> tensor<1x1x16xi1>
    %54 = tt.broadcast %53 : tensor<1x1x16xi1> -> tensor<1x8x16xi1>
    %55 = arith.andi %36, %54 : tensor<1x8x16xi1>
    %56 = tt.broadcast %55 : tensor<1x8x16xi1> -> tensor<4x8x16xi1>
    %57 = arith.andi %39, %56 : tensor<4x8x16xi1>
    tt.store %51, %42, %57 : tensor<4x8x16x!tt.ptr<f32>>
    tt.return
  }
}
