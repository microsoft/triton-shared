// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @rand(%arg0: !tt.ptr<i32, 1>, %arg1: !tt.ptr<i32, 1>) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<10x!tt.ptr<i32, 1>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i32, 1>>, tensor<10xi32>
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<10xi32>
    %4 = "tt.rand"(%3, %0) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xf32>
    %5 = tt.splat %arg1 : (!tt.ptr<i32, 1>) -> tensor<10x!tt.ptr<i32, 1>>
    %6 = tt.addptr %5, %0 : tensor<10x!tt.ptr<i32, 1>>, tensor<10xi32>
    %7 = arith.fptosi %4 : tensor<10xf32> to tensor<10xi32>
    tt.store %6, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<10xi32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @rand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<10xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<10xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_6_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : index to i32
// CHECK:             linalg.yield [[VAR_7_]] : i32
// CHECK:           } -> tensor<10xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [10], strides: [1] : memref<*xi32> to memref<10xi32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<10xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<10xi32, strided<[1]>> to memref<10xi32>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<10xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tt.rand"([[VAR_2_]], [[VAR_1_]]) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [10], strides: [1] : memref<*xi32> to memref<10xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<10xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]] : tensor<10xf32>) outs([[VAR_4_]] : tensor<10xi32>) {
// CHECK:           ^bb0([[in_:.+]]: f32, [[out_:.+]]: i32):
// CHECK:             [[VAR_6_1_:%.+]] = arith.fptosi [[in_]] : f32 to i32
// CHECK:             linalg.yield [[VAR_6_1_]] : i32
// CHECK:           } -> tensor<10xi32>
// CHECK:           memref.tensor_store [[VAR_5_]], [[VAR_reinterpret_cast_0_]] : memref<10xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
