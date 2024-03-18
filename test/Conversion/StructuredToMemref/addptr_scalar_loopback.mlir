// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: bf16
    tt.store %1, %10 : bf16
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[base_buffer_:%.+]], [[offset_:%.+]], [[sizes_:%.+]], [[VAR_strides_:%.+]] = memref.extract_strided_metadata [[VAR_reinterpret_cast_0_]] : memref<1xbf16, strided<[1], offset: ?>> -> memref<bf16>, index, index, index
// CHECK:           [[VAR_1_:%.+]] = arith.addi [[offset_]], [[VAR_0_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[base_buffer_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1], strides: [1] : memref<bf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[base_buffer_2_:%.+]], [[offset_3_:%.+]], [[sizes_4_:%.+]], [[VAR_strides_5_:%.+]] = memref.extract_strided_metadata [[VAR_reinterpret_cast_]] : memref<1xbf16, strided<[1], offset: ?>> -> memref<bf16>, index, index, index
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[offset_3_]], [[VAR_2_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[base_buffer_2_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1], strides: [1] : memref<bf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[LOAD_VAR_reinterpret_cast_1_MEM_:%.+]] = affine.load [[VAR_reinterpret_cast_1_]][0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:           affine.store [[LOAD_VAR_reinterpret_cast_1_MEM_]], [[VAR_reinterpret_cast_6_]][0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
