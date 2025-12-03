// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s
module {
  tt.func public @minmax_sgt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi sgt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_sgt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c-2147483648{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.maxsi %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ugt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c0{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.maxui %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_slt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c2147483647{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.minsi %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ult(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c-1{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.minui %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

// -----

module {
  tt.func public @nan_aware_max(%arg0: tensor<1024xf32>, %arg_out: !tt.ptr<f32>) {
    %res = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %cmp_gt = arith.cmpf ogt, %lhs, %rhs : f32
      %lhs_nan = arith.cmpf une, %lhs, %lhs : f32
      %pred = arith.ori %cmp_gt, %lhs_nan : i1
      %sel = arith.select %pred, %lhs, %rhs : f32
      tt.reduce.return %sel : f32
    }) : (tensor<1024xf32>) -> f32
    tt.store %arg_out, %res : !tt.ptr<f32>
    tt.return 
}
}

// CHECK-LABEL:  func.func @nan_aware_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1024xf32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_nan_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_nan_]] into [[VAR_0_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[PARAM_0_]] : tensor<1024xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: f32, [[in_]]it: f32) {
// CHECK:               [[CMP_gt_:%.+]] = arith.maximumf [[in_]], [[in_]]it : f32
// CHECK:               linalg.yield [[CMP_gt_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK:           tt.store [[PARAM_1_]], [[VAR_extracted_]] : !tt.ptr<f32>
// CHECK:           return
// CHECK:         }
