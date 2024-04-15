// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func public @bcast_kernel_01(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %6 = tt.splat %1 : i32 -> tensor<2048xi32>
    %7 = arith.addi %6, %5 : tensor<2048xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<32x!tt.ptr<f32, 1>>
    %9 = tt.addptr %8, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xf32>
    %11 = tt.reshape %10 {allow_reorder = false} : tensor<32xf32> -> tensor<1x32xf32>
    %12 = tt.broadcast %11 : tensor<1x32xf32> -> tensor<64x32xf32>
    %13 = tt.reshape %12 {allow_reorder = false} : tensor<64x32xf32> -> tensor<2048xf32>
    %14 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<2048x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %7 : tensor<2048x!tt.ptr<f32, 1>>, tensor<2048xi32>
    tt.store %15, %13 {cache = 1 : i32, evict = 1 : i32} : tensor<2048xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @bcast_kernel_01
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<[1, 32]> : tensor<2xi64>
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_5_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = tensor.reshape [[VAR_2_]]([[VAR_cst_]]) : (tensor<32xf32>, tensor<2xi64>) -> tensor<1x32xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tensor.empty() : tensor<64x32xf32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins([[VAR_reshape_]] : tensor<1x32xf32>) outs([[VAR_3_]] : tensor<64x32xf32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             linalg.yield [[IN_0_]] : f32
// CHECK:           } -> tensor<64x32xf32>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1xi64>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[CST_2048_]] : i64) outs([[VAR_5_]] : tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_reshape_0_:%.+]] = tensor.reshape [[VAR_4_]]([[VAR_6_]]) : (tensor<64x32xf32>, tensor<1xi64>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [2048], strides: [1] : memref<*xf32> to memref<2048xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_reshape_0_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<2048xf32>, memref<2048xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
