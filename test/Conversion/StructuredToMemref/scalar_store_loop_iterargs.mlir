// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg7 : i32 to index
    %1 = tt.addptr %arg1, %arg7 : !tt.ptr<f32>, i32
    %2 = arith.sitofp %arg7 : i32 to f32
    %3:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %1, %arg12 = %0) -> (!tt.ptr<f32>, index)  : i32 {
      tt.store %arg11, %2 : !tt.ptr<f32>
      %4 = tt.addptr %arg11, %arg10 : !tt.ptr<f32>, i32
      %5 = arith.index_cast %arg10 : i32 to index
      %6 = arith.addi %arg12, %5 : index
      scf.yield %4, %6 : !tt.ptr<f32>, index
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d1d2de3de
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32, [[PARAM_15_:%.+]]: i32) {
// CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = scf.for [[VAR_arg16_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]] iter_args([[VAR_arg17_:%.+]] = [[PARAM_7_]]) -> (i32)  : i32 {
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[VAR_arg17_]] : i32 to index
// CHECK:             [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             memref.store [[VAR_1_]], [[VAR_reinterpret_cast_]][%[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg17_]], [[VAR_arg16_]] : i32
// CHECK:             scf.yield [[VAR_6_]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
