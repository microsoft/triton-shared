// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save0 : tensor<1024x!tt.ptr<bf16>>,
    %save1 : tensor<1024x!tt.ptr<f32>>,
    %save2 : tensor<1024x!tt.ptr<f32>>,
    %save3 : tensor<1024x!tt.ptr<f32>>,
    %save4 : tensor<1024x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // f32ptr pointer
    %8 = tt.splat %f32ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // intptr pointer
    %18 = tt.splat %intptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    // f32ptr pointer
    %28 = tt.splat %f16ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %afm = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %aim = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi32>
    %bfm = tt.load %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf16>
    %5 = arith.truncf %afm : tensor<1024xf32> to tensor<1024xbf16>
    %6 = math.exp %afm : tensor<1024xf32>
    %7 = arith.sitofp %aim : tensor<1024xi32> to tensor<1024xf32>
    %10 = arith.extf %bfm : tensor<1024xf16> to tensor<1024xf32>
    %11 = math.sqrt %afm : tensor<1024xf32>
    tt.store %save0, %5 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xbf16>
    tt.store %save1, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %save2, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %save3, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %save4, %11 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: memref<*xf16>, [[PARAM_3_:%.+]]: tensor<1024x!tt.ptr<bf16, 1>>, [[PARAM_4_:%.+]]: tensor<1024x!tt.ptr<f32, 1>>, [[PARAM_5_:%.+]]: tensor<1024x!tt.ptr<f32, 1>>, [[PARAM_6_:%.+]]: tensor<1024x!tt.ptr<f32, 1>>, [[PARAM_7_:%.+]]: tensor<1024x!tt.ptr<f32, 1>>, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf16> to memref<1024xf16, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<1024xf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_1_]], [[RES_2_]] : memref<1024xf16, strided<[1]>> to memref<1024xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<1024xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = tensor.empty() : tensor<1024xbf16>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xf32>) outs([[VAR_3_]] : tensor<1024xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: bf16):
// CHECK:             [[VAR_11_:%.+]] = arith.truncf [[IN_0_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_11_]] : bf16
// CHECK:           } -> tensor<1024xbf16>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xf32>) outs([[VAR_0_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32):
// CHECK:             [[VAR_11_1_:%.+]] = math.exp [[IN_2_]] : f32
// CHECK:             linalg.yield [[VAR_11_1_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_1_]] : tensor<1024xi32>) outs([[VAR_6_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_11_2_:%.+]] = arith.sitofp [[IN_4_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_11_2_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_2_]] : tensor<1024xf16>) outs([[VAR_8_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_6_:%.+]]: f16, [[IN_7_:%.+]]: f32):
// CHECK:             [[VAR_11_3_:%.+]] = arith.extf [[IN_6_]] : f16 to f32
// CHECK:             linalg.yield [[VAR_11_3_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xf32>) outs([[VAR_0_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_8_:%.+]]: f32, [[IN_9_:%.+]]: f32):
// CHECK:             [[VAR_11_4_:%.+]] = math.sqrt [[IN_8_]] : f32
// CHECK:             linalg.yield [[VAR_11_4_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store [[PARAM_3_]], [[VAR_4_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xbf16>
// CHECK:           tt.store [[PARAM_4_]], [[VAR_5_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
// CHECK:           tt.store [[PARAM_5_]], [[VAR_7_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
// CHECK:           tt.store [[PARAM_6_]], [[VAR_9_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
// CHECK:           tt.store [[PARAM_7_]], [[VAR_10_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
// CHECK:           return
// CHECK:         }
