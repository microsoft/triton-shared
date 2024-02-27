// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = %arg1, offset = %1, size = 1, strides = 0
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : f32 -> tensor<1024xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %_ptr, %sum_out = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr_iter = %2, %sum_iter = %tensor_cf0) ->  (!tt.ptr<f32>, tensor<1024xf32>) {
      %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      // offset = 0, size = 1024, strides = 1
      %4 = tt.splat %ptr_iter : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      // source = %arg1, offset = %1, size = 1024, strides = 0
      %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // source = %arg1, offset = %1, size = 1024, strides = 1
      %8 = tt.load %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
      %9 = math.exp %8 : tensor<1024xf32>
      %sum_next = arith.addf %sum_iter, %9 : tensor<1024xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %ptr_next, %sum_next : !tt.ptr<f32>, tensor<1024xf32>
    }
    %10 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %21 = tt.addptr %20, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %21, %sum_out : tensor<1024xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK:           [[base_buffer_:%.+]], [[offset_:%.+]], [[sizes_:%.+]], [[VAR_strides_:%.+]] = memref.extract_strided_metadata [[VAR_reinterpret_cast_]] : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// CHECK:           [[VAR_5_:%.+]] = arith.addi [[offset_]], [[VAR_4_]] : index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[base_buffer_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = scf.for [[VAR_arg11_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg12_:%.+]] = [[VAR_reinterpret_cast_0_]], [[VAR_arg13_:%.+]] = [[VAR_1_]], [[VAR_arg14_:%.+]] = [[VAR_3_]]) -> (memref<1xf32, strided<[1], offset: ?>>, tensor<1024xf32>, index) {
// CHECK-DAG:         [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg14_]]{{.}}, sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_2_]], [[RES_]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK:             [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xf32>
// CHECK:             [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]] : tensor<1024xf32>) outs([[VAR_9_]] : tensor<1024xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:               [[VAR_14_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:               linalg.yield [[VAR_14_]] : f32
// CHECK:             } -> tensor<1024xf32>
// CHECK:             [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg13_]], [[VAR_10_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_arg13_]] : tensor<1024xf32>) {
// CHECK:             ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32):
// CHECK:               [[VAR_14_1_:%.+]] = arith.addf [[IN_2_]], [[IN_3_]] : f32
// CHECK:               linalg.yield [[VAR_14_1_]] : f32
// CHECK:             } -> tensor<1024xf32>
// CHECK:             [[base_buffer_3_:%.+]], [[offset_4_:%.+]], [[sizes_5_:%.+]], [[VAR_strides_6_:%.+]] = memref.extract_strided_metadata [[VAR_arg12_]] : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// CHECK:             [[VAR_12_:%.+]] = arith.addi [[offset_4_]], [[VAR_arg11_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[base_buffer_3_]] to offset: {{.}}[[VAR_12_]]{{.}}, sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_arg14_]], [[VAR_arg11_]] : index
// CHECK:             scf.yield [[VAR_reinterpret_cast_7_]], [[VAR_11_]], [[VAR_13_]] : memref<1xf32, strided<[1], offset: ?>>, tensor<1024xf32>, index
// CHECK:           }
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_8_]]{{.}}, sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_6_]]#1 in writable [[VAR_reinterpret_cast_1_]] : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
