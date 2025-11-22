// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s

module {
  tt.func public @addi(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.addi %arg14, %arg15 : i32
      tt.reduce.return %69 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL:   func.func @addi(
// CHECK-SAME:                    %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:                    %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[ALLOC_TENSOR_0:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[CONSTANT_0]] into %[[ALLOC_TENSOR_0]][] : tensor<i32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xi32>) outs(%[[INSERT_0]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:               linalg.yield %[[ADDI_0]] : i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<i32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
