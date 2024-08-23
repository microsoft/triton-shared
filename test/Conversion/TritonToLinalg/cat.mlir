// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<128x!tt.ptr<f64>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f64>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f64>>
    %4 = tt.splat %arg1 : !tt.ptr<f64> -> tensor<128x!tt.ptr<f64>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f64>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<f64>>
    %7 = tt.cat %3, %6 : tensor<128xf64> -> tensor<256xf64>
    %8 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %9 = tt.splat %arg2 : !tt.ptr<f64> -> tensor<256x!tt.ptr<f64>>
    %10 = tt.addptr %9, %8 : tensor<256x!tt.ptr<f64>>, tensor<256xi32>
    tt.store %10, %7 : tensor<256x!tt.ptr<f64>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xf64> {tt.divisibility = 16 : i32}, %arg1: memref<*xf64> {tt.divisibility = 16 : i32}, %arg2: memref<*xf64> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     [[REINTERPRET_CAST_:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xf64> to memref<128xf64, strided<[1]>>
// CHECK:     [[ALLOC_:%.+]] = memref.alloc() : memref<128xf64>
// CHECK:     memref.copy [[REINTERPRET_CAST_]], [[ALLOC_]] : memref<128xf64, strided<[1]>> to memref<128xf64>
// CHECK:     [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<128xf64>
// CHECK:     [[REINTERPRET_CAST_0_:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xf64> to memref<128xf64, strided<[1]>>
// CHECK:     [[ALLOC_1_:%.+]] = memref.alloc() : memref<128xf64>
// CHECK:     memref.copy [[REINTERPRET_CAST_0_]], [[ALLOC_1_]] : memref<128xf64, strided<[1]>> to memref<128xf64>
// CHECK:     [[VAR_1_:%.+]] = bufferization.to_tensor [[ALLOC_1_]] restrict writable : memref<128xf64>
// CHECK:     [[VAR_2_:%.+]] = tensor.empty() : tensor<256xf64>
// CHECK:     [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (-d0 + 127)>], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<128xf64>) outs([[VAR_2_]] : tensor<256xf64>) {
// CHECK:     ^bb0([[IN_:%.+]]: f64, [[OUT_:%.+]]: f64):
// CHECK:       linalg.yield [[IN_]] : f64
// CHECK:     } -> tensor<256xf64>
// CHECK:     [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0 + 128)>], iterator_types = ["parallel"]} ins([[VAR_1_]] : tensor<128xf64>) outs([[VAR_3_]] : tensor<256xf64>) {
// CHECK:     ^bb0([[IN_1_:%.+]]: f64, [[OUT_1_:%.+]]: f64):
// CHECK:       linalg.yield [[IN_1_]] : f64
// CHECK:     } -> tensor<256xf64>
// CHECK:     [[REINTERPRET_CAST_2_:%.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [256], strides: [1]{{.*}} : memref<*xf64> to memref<256xf64, strided<[1]>>
// CHECK:     bufferization.materialize_in_destination [[VAR_4_]] in writable [[REINTERPRET_CAST_2_]] : (tensor<256xf64>, memref<256xf64, strided<[1]>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
