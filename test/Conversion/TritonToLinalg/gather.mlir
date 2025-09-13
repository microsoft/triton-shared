// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @gather_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<4x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<4x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = tt.broadcast %2 : tensor<4x1xi32> -> tensor<4x4xi32>
    %5 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<4x4xi32>
    %6 = arith.addi %4, %5 : tensor<4x4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %9 = tt.load %8 : tensor<4x4x!tt.ptr<f32>>
    %10 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %12 = arith.muli %11, %cst : tensor<8x1xi32>
    %13 = tt.broadcast %12 : tensor<8x1xi32> -> tensor<8x4xi32>
    %14 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<8x4xi32>
    %15 = arith.addi %13, %14 : tensor<8x4xi32>
    %16 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x4x!tt.ptr<i64>>
    %17 = tt.addptr %16, %15 : tensor<8x4x!tt.ptr<i64>>, tensor<8x4xi32>
    %18 = tt.load %17 : tensor<8x4x!tt.ptr<i64>>
    %19 = tt.gather %9[%18] {axis = 0 : i32} : (tensor<4x4xf32>, tensor<8x4xi64>) -> tensor<8x4xf32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %15 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %21, %19 : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @gather_test_kernel(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                  %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                  %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                  %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                  %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                  %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_10:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_11:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [4, 4], strides: {{\[}}%[[VAL_10]], 1] : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK:           memref.copy %[[VAL_11]], %[[VAL_12]] : memref<4x4xf32, strided<[?, 1]>> to memref<4x4xf32>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_tensor %[[VAL_12]] restrict writable : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:           %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [8, 4], strides: {{\[}}%[[VAL_10]], 1] : memref<*xi64> to memref<8x4xi64, strided<[?, 1]>>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() : memref<8x4xi64>
// CHECK:           memref.copy %[[VAL_14]], %[[VAL_15]] : memref<8x4xi64, strided<[?, 1]>> to memref<8x4xi64>
// CHECK:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_15]] restrict writable : memref<8x4xi64> to tensor<8x4xi64>
// CHECK:           %[[VAL_17:.*]] = tensor.empty() : tensor<8x4xf32>
// CHECK:           %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_16]] : tensor<8x4xi64>) outs(%[[VAL_17]] : tensor<8x4xf32>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: i64, %[[VAL_20:.*]]: f32):
// CHECK:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_19]] : i64 to index
// CHECK:             %[[VAL_22:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_23:.*]] = arith.index_cast %[[VAL_19]] : i64 to index
// CHECK:             %[[VAL_24:.*]] = arith.cmpi slt, %[[VAL_23]], %[[VAL_10]] : index
// CHECK:             cf.assert %[[VAL_24]], "index must be smaller than axis size"
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sge, %[[VAL_19]], %[[VAL_9]] : i64
// CHECK:             cf.assert %[[VAL_25]], "index must be larger or equal to 0"
// CHECK:             %[[VAL_26:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_21]], %[[VAL_22]]] : tensor<4x4xf32>
// CHECK:             linalg.yield %[[VAL_26]] : f32
// CHECK:           } -> tensor<8x4xf32>
// CHECK:           %[[VAL_27:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [8, 4], strides: {{\[}}%[[VAL_10]], 1] : memref<*xf32> to memref<8x4xf32, strided<[?, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_18]] in writable %[[VAL_27]] : (tensor<8x4xf32>, memref<8x4xf32, strided<[?, 1]>>) -> ()
// CHECK:           return
// CHECK:         }
