// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
tt.func public @assert_lol(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
  %1 = tt.splat %0 : (i1) -> tensor<1xi1>
  tt.assert %1, "lol", "", "", 0 : tensor<1xi1>
  tt.return
}

// CHECK-LABEL:  func.func @assert_lol
// CHECK-SAME:   ([[PARAM_0_:%.+]]: i32, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.cmpi sgt, [[PARAM_0_]], [[CST_0_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<1xi1>
// CHECK:           [[VAR_2_:%.+]] = linalg.fill ins([[VAR_0_]] : i1) outs([[VAR_1_]] : tensor<1xi1>) -> tensor<1xi1>
// CHECK:           cf.assert [[VAR_0_]], ".py:0:  Assertion `lol` failed"
// CHECK:           return
// CHECK:         }
