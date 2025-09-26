// RUN: triton-shared-opt --split-input-file --structured-to-memref %s | FileCheck %s

module {
  tt.func public @wrap_stacked_load(%arg0: !tt.ptr<f32>, %M: index, %N: index) -> tensor<4x4xf32> {
    %0 = arith.muli %M, %N : index
    %1 = tts.make_tptr %arg0 to sizes: [4, 4], strides: [%N, 1], offsets: [0, 0], shape: [%0, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x4x!tt.ptr<f32>>) -> tensor<4x4xf32>
    tt.return %2 : tensor<4x4xf32>
  }
}

// CHECK-LABEL:   tt.func public @wrap_stacked_load(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:                                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> tensor<4x4xf32> {
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_5]], %[[VAL_2]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_4]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_5]] : index
// CHECK:           %[[VAL_12:.*]] = arith.divsi %[[VAL_11]], %[[VAL_2]] : index
// CHECK:           %[[VAL_13:.*]] = arith.minsi %[[VAL_12]], %[[VAL_6]] : index
// CHECK:           %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_5]]], sizes: {{\[}}%[[VAL_13]], %[[VAL_7]]], strides: {{\[}}%[[VAL_2]], %[[VAL_8]]] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_15:.*]] = arith.subi %[[VAL_6]], %[[VAL_13]] : index
// CHECK:           %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_9]]], sizes: {{\[}}%[[VAL_15]], %[[VAL_7]]], strides: {{\[}}%[[VAL_2]], %[[VAL_8]]] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_17:.*]] = builtin.unrealized_conversion_cast %[[VAL_14]], %[[VAL_16]] : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>> to tensor<4x4x!tt.ptr<f32>> {wrap_stacked}
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_22:.*]] = memref.dim %[[VAL_14]], %[[VAL_21]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_24:.*]] = memref.dim %[[VAL_14]], %[[VAL_23]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_26:.*]] = memref.dim %[[VAL_16]], %[[VAL_25]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_28:.*]] = memref.dim %[[VAL_16]], %[[VAL_27]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_29:.*]] = memref.subview %[[VAL_18]]{{\[}}%[[VAL_19]], %[[VAL_19]]] {{\[}}%[[VAL_22]], %[[VAL_24]]] {{\[}}%[[VAL_20]], %[[VAL_20]]] : memref<4x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_30:.*]] = memref.subview %[[VAL_18]]{{\[}}%[[VAL_22]], %[[VAL_19]]] {{\[}}%[[VAL_26]], %[[VAL_28]]] {{\[}}%[[VAL_20]], %[[VAL_20]]] : memref<4x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           memref.copy %[[VAL_14]], %[[VAL_29]] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           memref.copy %[[VAL_16]], %[[VAL_30]] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_31:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:           tt.return %[[VAL_31]] : tensor<4x4xf32>
// CHECK:         }
