// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func public @unsplat_kernel(%arg0: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<42> : tensor<1xi32>
    %c42_i32 = arith.constant 42 : i32
    %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>>
    %2 = arith.cmpi sgt, %1, %cst : tensor<1xi32>
    %3 = tt.unsplat %2 : tensor<1xi1>
    scf.if %3 {
      tt.store %arg0, %c42_i32 : !tt.ptr<i32>
    }
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @unsplat_kernel(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[CAST_0]] restrict : memref<?xi32> to tensor<?xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_1]] : tensor<1xi32>) outs(%[[EMPTY_0]] : tensor<1xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[INDEX_CAST_0]]] : tensor<?xi32>
// CHECK:             linalg.yield %[[EXTRACT_0]] : i32
// CHECK:           } -> tensor<1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1xi1>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<1xi32>, tensor<1xi32>) outs(%[[EMPTY_1]] : tensor<1xi1>) {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i1):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:             linalg.yield %[[CMPI_0]] : i1
// CHECK:           } -> tensor<1xi1>
// CHECK:           %[[EXTRACT_1:.*]] = tensor.extract %[[GENERIC_1]]{{\[}}%[[CONSTANT_2]]] : tensor<1xi1>
// CHECK:           scf.if %[[EXTRACT_1]] {
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:             affine.store %[[CONSTANT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
