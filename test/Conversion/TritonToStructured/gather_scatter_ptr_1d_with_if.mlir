// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// Make sure tts.make_gather_scatter_tptr is generated with for 1D tensor on addptr with if.

// CHECK-LABEL:   tt.func public @gather_row(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                               %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<8> : tensor<4xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<4> : tensor<4xi32>
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = arith.divsi %[[VAL_7]], %[[VAL_6]] : tensor<4xi32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_10:.*]] = scf.if %[[VAL_9]] -> (tensor<4xi32>) {
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_4]] : tensor<4xi32>
// CHECK:             scf.yield %[[VAL_11]] : tensor<4xi32>
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_8]] : tensor<4xi32>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_10]], strides: [1], offsets: [0] : tensor<4xi32> <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:           %[[VAL_13:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_10]], strides: [1], offsets: [0] : tensor<4xi32> <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:           %[[VAL_14:.*]] = "tts.load"(%[[VAL_12]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>) -> tensor<4xf32>
// CHECK:           "tts.store"(%[[VAL_13]], %[[VAL_14]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>, tensor<4xf32>) -> ()
// CHECK:           tt.retur

module {
  tt.func public @gather_row(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<4xi32>
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<4xi32>
    %2 = arith.cmpi slt, %arg2, %c8_i32 : i32
    %3 = scf.if %2 -> (tensor<4xi32>) {
      %9 = arith.addi %1, %cst : tensor<4xi32>
      scf.yield %9 : tensor<4xi32>
    } else {
      scf.yield %1 : tensor<4xi32>
    }
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %7 = tt.addptr %6, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %8 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    tt.store %7, %8 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
