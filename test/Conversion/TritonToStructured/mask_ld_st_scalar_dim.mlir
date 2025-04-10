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

// CHECK-LABEL:   tt.func @mask_ld_st_scalar(
// CHECK:                               %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [2, 1], strides: {{\[}}%[[VAL_9]], 0], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x1x!tt.ptr<f32>>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_12:.*]] = arith.minsi %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = arith.maxsi %[[VAL_12]], %[[VAL_7]] : index
// CHECK:           %[[VAL_14:.*]] = arith.minsi %[[VAL_13]], %[[VAL_8]] : index
// CHECK:           %[[VAL_15:.*]] = "tts.load"(%[[VAL_10]], %[[VAL_14]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, index) -> tensor<2x1xf32>
// CHECK:           %[[VAL_16:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_17:.*]] = tts.make_tptr %[[VAL_0]] to sizes: [2, 1], strides: {{\[}}%[[VAL_16]], 0], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x1x!tt.ptr<f32>>
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = arith.minsi %[[VAL_18]], %[[VAL_8]] : index
// CHECK:           %[[VAL_20:.*]] = arith.maxsi %[[VAL_19]], %[[VAL_7]] : index
// CHECK:           %[[VAL_21:.*]] = arith.minsi %[[VAL_20]], %[[VAL_8]] : index
// CHECK:           %[[VAL_22:.*]] = "tts.load"(%[[VAL_17]], %[[VAL_21]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, index) -> tensor<2x1xf32>
// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_24:.*]] = arith.minsi %[[VAL_23]], %[[VAL_8]] : index
// CHECK:           %[[VAL_25:.*]] = arith.maxsi %[[VAL_24]], %[[VAL_7]] : index
// CHECK:           %[[VAL_26:.*]] = arith.minsi %[[VAL_25]], %[[VAL_8]] : index
// CHECK:           "tts.store"(%[[VAL_10]], %[[VAL_15]], %[[VAL_26]]) <{static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, tensor<2x1xf32>, index) -> ()
// CHECK:           %[[VAL_27:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_28:.*]] = arith.minsi %[[VAL_27]], %[[VAL_8]] : index
// CHECK:           %[[VAL_29:.*]] = arith.maxsi %[[VAL_28]], %[[VAL_7]] : index
// CHECK:           %[[VAL_30:.*]] = arith.minsi %[[VAL_29]], %[[VAL_8]] : index
// CHECK:           "tts.store"(%[[VAL_17]], %[[VAL_22]], %[[VAL_30]]) <{static_mask_dims = array<i64: -9223372036854775808, 1>}> : (tensor<2x1x!tt.ptr<f32>>, tensor<2x1xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
