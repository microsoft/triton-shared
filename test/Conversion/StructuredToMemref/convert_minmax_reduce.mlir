// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref  %s | FileCheck %s
module {
  tt.func public @minmax_sgt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi sgt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : i32
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_sgt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_2147483648_:%.+]] = arith.constant -2147483648 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_0_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_minus_2147483648_]] into [[VAR_2_]][] : tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xi32>) outs([[VAR_inserted_]] : tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: i32, [[init_:%.+]]: i32) {
// CHECK:               [[VAR_3_:%.+]] = arith.maxsi [[in_]], [[init_]] : i32
// CHECK:               linalg.yield [[VAR_3_]] : i32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<i32>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }


// -----

module {
  tt.func public @minmax_ugt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ugt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : i32
    tt.return
  }
}


// CHECK-LABEL:  func.func @minmax_ugt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_0_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_]] into [[VAR_2_]][] : tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xi32>) outs([[VAR_inserted_]] : tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: i32, [[init_:%.+]]: i32) {
// CHECK:               [[VAR_3_:%.+]] = arith.maxui [[in_]], [[init_]] : i32
// CHECK:               linalg.yield [[VAR_3_]] : i32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<i32>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }


// -----

module {
  tt.func public @minmax_slt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi slt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : i32
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_slt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2147483647_:%.+]] = arith.constant 2147483647 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_0_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_2147483647_]] into [[VAR_2_]][] : tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xi32>) outs([[VAR_inserted_]] : tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: i32, [[init_:%.+]]: i32) {
// CHECK:               [[VAR_3_:%.+]] = arith.minsi [[in_]], [[init_]] : i32
// CHECK:               linalg.yield [[VAR_3_]] : i32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<i32>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }



// -----

module {
  tt.func public @minmax_ult(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ult, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : i32
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_ult
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_0_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_minus_1_]] into [[VAR_2_]][] : tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xi32>) outs([[VAR_inserted_]] : tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: i32, [[init_:%.+]]: i32) {
// CHECK:               [[VAR_3_:%.+]] = arith.minui [[in_]], [[init_]] : i32
// CHECK:               linalg.yield [[VAR_3_]] : i32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<i32>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
