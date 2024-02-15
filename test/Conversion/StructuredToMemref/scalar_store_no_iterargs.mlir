// RUN: triton-shared-opt --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s

module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.addptr %arg1, %arg7 : !tt.ptr<f32, 1>, i32
    %1 = arith.sitofp %arg7 : i32 to f32
    scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %2 = tt.addptr %0, %arg10 : !tt.ptr<f32, 1>, i32
      tt.store %2, %1 {cache = 1 : i32, evict = 1 : i32} : f32
    }
    return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d1d2de3de
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:           [[base_buffer_:%.+]], [[offset_:%.+]], [[sizes_:%.+]], [[VAR_strides_:%.+]] = memref.extract_strided_metadata [[VAR_reinterpret_cast_]] : memref<1xf32> -> memref<f32>, index, index, index
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[base_buffer_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]]  : i32 {
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[I_0_]] : i32 to index
// CHECK:             [[base_buffer_1_:%.+]], [[offset_2_:%.+]], [[sizes_3_:%.+]], [[VAR_strides_4_:%.+]] = memref.extract_strided_metadata [[VAR_reinterpret_cast_0_]] : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[offset_2_]], [[VAR_2_]] : index
// CHECK:             [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[base_buffer_1_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store [[VAR_1_]], [[VAR_reinterpret_cast_5_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
