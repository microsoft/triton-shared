// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
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
    return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d1d2de3de
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK:           [[VAR_2_:%.+]]:3 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg12_:%.+]] = [[VAR_0_]], [[VAR_arg13_:%.+]] = [[VAR_0_]]) -> (memref<1xf32, strided<[1], offset: ?>>, index, index)  : i32 {
// CHECK:             affine.store [[VAR_1_]], [[VAR_arg11_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_arg10_]] : i32 to index
// CHECK:             [[base_buffer_:%.+]], [[offset_:%.+]], [[sizes_:%.+]], [[VAR_strides_:%.+]] = memref.extract_strided_metadata [[VAR_arg11_]] : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[offset_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[base_buffer_]] to offset: {{.}}[[VAR_4_]]{{.}}, sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg13_]], [[VAR_3_]] : index
// CHECK:             scf.yield [[VAR_reinterpret_cast_0_]], [[VAR_5_]], [[VAR_6_]] : memref<1xf32, strided<[1], offset: ?>>, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @reduce_kernel_2d_0d1d2de3de
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK:           [[VAR_2_:%.+]]:4 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg12_:%.+]] = [[VAR_0_]], [[VAR_arg13_:%.+]] = [[VAR_0_]], [[VAR_arg14_:%.+]] = [[VAR_0_]]) -> (memref<1xf32, strided<[1], offset: ?>>, index, index, index)  : i32 {
// CHECK:             affine.store [[VAR_1_]], [[VAR_arg11_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_arg10_]] : i32 to index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[VAR_arg11_]] to offset: {{.}}[[VAR_4_]]{{.}}, sizes: [1], strides: [1] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg13_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg14_]], [[VAR_3_]] : index
// CHECK:             scf.yield [[VAR_reinterpret_cast_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]] : memref<1xf32, strided<[1], offset: ?>>, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
