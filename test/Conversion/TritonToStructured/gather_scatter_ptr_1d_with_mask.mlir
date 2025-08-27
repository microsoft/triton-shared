// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// Make sure tts.load is generated with correct mask for 1D tensor.

// CHECK-LABEL:   tt.func public @row_gather1d_with_mask(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:                                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_6:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<32x!tt.ptr<i32>>
// CHECK:           %[[VAL_7:.*]] = "tts.load"(%[[VAL_6]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<i32>>) -> tensor<32xi32>
// CHECK:           %[[VAL_8:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [32] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_7]], strides: [1], offsets: [0] : tensor<32xi32> <f32> to !tt.ptr<tensor<32xf32>>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = arith.minsi %[[VAL_9]], %[[VAL_5]] : index
// CHECK:           %[[VAL_11:.*]] = arith.maxsi %[[VAL_10]], %[[VAL_4]] : index
// The tts.load with mask.
// CHECK:           %[[VAL_12:.*]] = "tts.load"(%[[VAL_8]], %[[VAL_11]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (!tt.ptr<tensor<32xf32>>, index) -> tensor<32xf32>
// CHECK:           %[[VAL_13:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[VAL_13]], %[[VAL_12]]) <{static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>) -> ()
// CHECK:           tt.return

module attributes {} {
  tt.func public @row_gather1d_with_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<i32>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %7 = arith.cmpi slt, %0, %6 : tensor<32xi32>
    %8 = tt.load %5, %7: tensor<32x!tt.ptr<f32>>
    %9 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %10 = tt.addptr %9, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %10, %8: tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
