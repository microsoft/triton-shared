// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// Make sure make_gather_scatter_tptr with generic mask generate correctly.

// CHECK-LABEL:   tt.func public @generic_mask_3d_non_continuous_load_kernel(
// CHECK-SAME:                                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                                %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                                                %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_11:.*]] = arith.constant -2.000000e+00 : f32
// CHECK:           %[[VAL_12:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_16:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_17:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<4xi32>
// CHECK:           %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_17]] : tensor<4xi32>
// CHECK:           %[[VAL_19:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
// CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_21:.*]] = arith.minsi %[[VAL_20]], %[[VAL_15]] : index
// CHECK:           %[[VAL_22:.*]] = arith.maxsi %[[VAL_21]], %[[VAL_14]] : index
// CHECK:           %[[VAL_23:.*]] = "tts.load"(%[[VAL_19]], %[[VAL_22]], %[[VAL_13]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<i32>>, index, i32) -> tensor<4xi32>
// CHECK:           %[[VAL_24:.*]] = tt.splat %[[VAL_5]] : i32 -> tensor<4xi32>
// CHECK:           %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_23]], %[[VAL_24]] : tensor<4xi32>
// CHECK:           %[[VAL_26:.*]] = arith.andi %[[VAL_25]], %[[VAL_18]] : tensor<4xi1>
// CHECK:           %[[VAL_27:.*]] = arith.index_cast %[[VAL_7]] : i32 to index
// CHECK:           %[[VAL_28:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_30:.*]] = arith.minsi %[[VAL_29]], %[[VAL_15]] : index
// CHECK:           %[[VAL_31:.*]] = arith.maxsi %[[VAL_30]], %[[VAL_14]] : index
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_33:.*]] = arith.minsi %[[VAL_32]], %[[VAL_12]] : index
// CHECK:           %[[VAL_34:.*]] = arith.maxsi %[[VAL_33]], %[[VAL_14]] : index
// CHECK:           %[[VAL_35:.*]] = arith.minsi %[[VAL_34]], %[[VAL_12]] : index
// CHECK:           %[[VAL_36:.*]] = arith.minsi %[[VAL_31]], %[[VAL_15]] : index
// CHECK:           %[[VAL_37:.*]] = arith.minsi %[[VAL_35]], %[[VAL_12]] : index
// CHECK:           %[[VAL_38:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4, 4, 16] gather_scatter_dim: 1 gather_scatter_offset: %[[VAL_23]] gather_scatter_mask: %[[VAL_26]], strides: {{\[}}%[[VAL_27]], %[[VAL_28]], 1], offsets: [0, 0, 0] : tensor<4xi32> tensor<4xi1> <f32> to !tt.ptr<tensor<4x4x16xf32>>
// CHECK:           %[[VAL_39:.*]] = "tts.load"(%[[VAL_38]], %[[VAL_36]], %[[VAL_37]], %[[VAL_11]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, 0, -9223372036854775808>}> : (!tt.ptr<tensor<4x4x16xf32>>, index, index, f32) -> tensor<4x4x16xf32>
// CHECK:           %[[VAL_40:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
// CHECK:           %[[VAL_41:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:           %[[VAL_42:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [4, 4, 16], strides: {{\[}}%[[VAL_40]], %[[VAL_41]], 1], offsets: [0, 0, 0], shape: [0, 0, 0], order: [] : <f32> to tensor<4x4x16x!tt.ptr<f32>>
// CHECK:           %[[VAL_43:.*]] = arith.minsi %[[VAL_22]], %[[VAL_15]] : index
// CHECK:           %[[VAL_44:.*]] = arith.minsi %[[VAL_43]], %[[VAL_15]] : index
// CHECK:           "tts.store"(%[[VAL_42]], %[[VAL_39]], %[[VAL_36]], %[[VAL_44]], %[[VAL_37]]) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (tensor<4x4x16x!tt.ptr<f32>>, tensor<4x4x16xf32>, index, index, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @generic_mask_3d_non_continuous_load_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-2.000000e+00> : tensor<4x4x16xf32>
    %cst_0 = arith.constant dense<0> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %3 = arith.cmpi slt, %0, %2 : tensor<4xi32>
    %4 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %6 = tt.load %5, %3, %cst_0 : tensor<4x!tt.ptr<i32>>
    %7 = tt.splat %arg5 : i32 -> tensor<4xi32>
    %8 = arith.cmpi slt, %6, %7 : tensor<4xi32>
    %9 = arith.andi %8, %3 : tensor<4xi1>
    %10 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %11 = tt.expand_dims %10 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %12 = tt.splat %arg7 : i32 -> tensor<4x1x1xi32>
    %13 = arith.muli %11, %12 : tensor<4x1x1xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %16 = tt.expand_dims %6 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %17 = tt.expand_dims %16 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %18 = tt.splat %arg8 : i32 -> tensor<1x4x1xi32>
    %19 = arith.muli %17, %18 : tensor<1x4x1xi32>
    %20 = tt.broadcast %15 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x4x1x!tt.ptr<f32>>
    %21 = tt.broadcast %19 : tensor<1x4x1xi32> -> tensor<4x4x1xi32>
    %22 = tt.addptr %20, %21 : tensor<4x4x1x!tt.ptr<f32>>, tensor<4x4x1xi32>
    %23 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %25 = tt.broadcast %22 : tensor<4x4x1x!tt.ptr<f32>> -> tensor<4x4x16x!tt.ptr<f32>>
    %26 = tt.broadcast %24 : tensor<1x1x16xi32> -> tensor<4x4x16xi32>
    %27 = tt.addptr %25, %26 : tensor<4x4x16x!tt.ptr<f32>>, tensor<4x4x16xi32>
    %28 = tt.splat %arg4 : i32 -> tensor<4x1x1xi32>
    %29 = arith.cmpi slt, %11, %28 : tensor<4x1x1xi32>
    %30 = tt.expand_dims %9 {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %31 = tt.expand_dims %30 {axis = 2 : i32} : tensor<1x4xi1> -> tensor<1x4x1xi1>
    %32 = tt.splat %arg6 : i32 -> tensor<1x1x16xi32>
    %33 = arith.cmpi slt, %24, %32 : tensor<1x1x16xi32>
    %34 = tt.broadcast %31 : tensor<1x4x1xi1> -> tensor<1x4x16xi1>
    %35 = tt.broadcast %33 : tensor<1x1x16xi1> -> tensor<1x4x16xi1>
    %36 = arith.andi %34, %35 : tensor<1x4x16xi1>
    %37 = tt.broadcast %29 : tensor<4x1x1xi1> -> tensor<4x4x16xi1>
    %38 = tt.broadcast %36 : tensor<1x4x16xi1> -> tensor<4x4x16xi1>
    %39 = arith.andi %37, %38 : tensor<4x4x16xi1>
    %40 = tt.load %27, %39, %cst : tensor<4x4x16x!tt.ptr<f32>>
    %41 = tt.splat %arg9 : i32 -> tensor<4x1x1xi32>
    %42 = arith.muli %11, %41 : tensor<4x1x1xi32>
    %43 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x1x!tt.ptr<f32>>
    %44 = tt.addptr %43, %42 : tensor<4x1x1x!tt.ptr<f32>>, tensor<4x1x1xi32>
    %45 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %46 = tt.expand_dims %45 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %47 = tt.splat %arg10 : i32 -> tensor<1x4x1xi32>
    %48 = arith.muli %46, %47 : tensor<1x4x1xi32>
    %49 = tt.broadcast %44 : tensor<4x1x1x!tt.ptr<f32>> -> tensor<4x4x1x!tt.ptr<f32>>
    %50 = tt.broadcast %48 : tensor<1x4x1xi32> -> tensor<4x4x1xi32>
    %51 = tt.addptr %49, %50 : tensor<4x4x1x!tt.ptr<f32>>, tensor<4x4x1xi32>
    %52 = tt.broadcast %51 : tensor<4x4x1x!tt.ptr<f32>> -> tensor<4x4x16x!tt.ptr<f32>>
    %53 = tt.addptr %52, %26 : tensor<4x4x16x!tt.ptr<f32>>, tensor<4x4x16xi32>
    %54 = tt.splat %arg3 : i32 -> tensor<1x4x1xi32>
    %55 = arith.cmpi slt, %46, %54 : tensor<1x4x1xi32>
    %56 = tt.broadcast %55 : tensor<1x4x1xi1> -> tensor<1x4x16xi1>
    %57 = arith.andi %56, %35 : tensor<1x4x16xi1>
    %58 = tt.broadcast %57 : tensor<1x4x16xi1> -> tensor<4x4x16xi1>
    %59 = arith.andi %37, %58 : tensor<4x4x16xi1>
    tt.store %53, %40, %59 : tensor<4x4x16x!tt.ptr<f32>>
    tt.return
  }
}