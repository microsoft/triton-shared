// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<i32>,
    %b : !tt.ptr<i32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<i32>>
        %5 = arith.addi %am, %bm : tensor<1024xi32>
        tt.store %19, %5 : tensor<1024x!tt.ptr<i32>>
        tt.return
    }
}
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]], [[VAR_1_]] : tensor<1024xi32>, tensor<1024xi32>) outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32):
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[IN_0_]], [[IN_1_]] : i32
// CHECK:             linalg.yield [[VAR_3_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024xi32>, memref<1024xi32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_0_]]{{.}}, sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[CST_0_]]{{.}}, sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]], [[VAR_1_]] : tensor<1024xi32>, tensor<1024xi32>) outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32):
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[IN_0_]], [[IN_1_]] : i32
// CHECK:             linalg.yield [[VAR_3_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024xi32>, memref<1024xi32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
