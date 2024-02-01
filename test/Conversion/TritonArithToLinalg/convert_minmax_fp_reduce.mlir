// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @maxnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.maxnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : f32
    tt.return
  }
}


// -----


module {
  tt.func public @minnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.minnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 {cache = 1 : i32, evict = 1 : i32} : f32
    tt.return
  }
}

// CHECK-LABEL:  func.func @maxnumf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32, 1>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: f32, [[in_]]it: f32) {
// CHECK:               [[VAR_3_:%.+]] = arith.maxnumf [[in_]], [[in_]]it : f32
// CHECK:               linalg.yield [[VAR_3_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK:           tt.store [[PARAM_0_]], [[VAR_extracted_]] {cache = 1 : i32, evict = 1 : i32} : f32
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:  func.func @minnumf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32, 1>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: f32, [[in_]]it: f32) {
// CHECK:               [[VAR_3_:%.+]] = arith.minnumf [[in_]], [[in_]]it : f32
// CHECK:               linalg.yield [[VAR_3_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK:           tt.store [[PARAM_0_]], [[VAR_extracted_]] {cache = 1 : i32, evict = 1 : i32} : f32
// CHECK:           return
// CHECK:         }
