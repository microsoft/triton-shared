// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<1024x!tt.ptr<f32>> -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1], strides = [0, 0]
    %5 = tt.broadcast %4 : tensor<1024x1x!tt.ptr<f32>> -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1024], strides = [0, 0]
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    // offset = [0, 0], size = [1, 1024], strides = [0, 1]
    %8 = tt.broadcast %7 : tensor<1x1024xi32> -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [0, 1]
    %9 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<1024xi32> -> tensor<1024x1xi32>
    // offset = [0, 0], size = [1024, 1], strides = [1, 0]
    %11 = tt.broadcast %10 : tensor<1024x1xi32> -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [1, 0]
    %12 = arith.addi %8, %11 : tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [1, 1]
    %13 = tt.addptr %5, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg1, offset = [pid * %arg2, 0], size = [1024, 1024], strides = [1, 1]
    %14 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x1024xf32>
    %17 = math.exp %14 : tensor<1024x1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = pid+arg3, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = pid+arg3, size = 1024, strides = 0
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<1024x!tt.ptr<f32>> -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1], strides = [0, 0]
    %22 = tt.broadcast %21 : tensor<1024x1x!tt.ptr<f32>> -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [0, 0]
    %23 = tt.addptr %22, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [1, 1]
    tt.store %23, %17 : tensor<1024x1024xf32>
    tt.return
  }
}
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1024, 1024], strides: [1, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[1, 1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024x1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024x1024xf32, strided<[1, 1], offset: ?>> to memref<1024x1024xf32>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024x1024xf32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_2_]] : tensor<1024x1024xf32>) outs([[VAR_2_]] : tensor<1024x1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_6_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_6_]] : f32
// CHECK:           } -> tensor<1024x1024xf32>
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: [1024, 1024], strides: [1, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[1, 1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_3_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024x1024xf32>, memref<1024x1024xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
