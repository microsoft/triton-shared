// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %3 = tt.load %2 : tensor<256x!tt.ptr<i32>>
    %4 = tt.reshape %3 {allow_reorder = false} : tensor<256xi32> -> tensor<128x2xi32>
    %outLHS, %outRHS = tt.split %4 : tensor<128x2xi32> -> tensor<128xi32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %7 = tt.addptr %6, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %7, %outLHS : tensor<128x!tt.ptr<i32>>
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %9 = tt.addptr %8, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %9, %outRHS : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     %cst = arith.constant dense<[128, 2]> : tensor<2xi64>
// CHECK:     [[REINTERPRET_CAST_:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1]{{.*}} : memref<*xi32> to memref<256xi32, strided<[1]>>
// CHECK:     [[ALLOC_:%.+]] = memref.alloc() : memref<256xi32>
// CHECK:     memref.copy [[REINTERPRET_CAST_]], [[ALLOC_]] : memref<256xi32, strided<[1]>> to memref<256xi32>
// CHECK:     [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<256xi32>
// CHECK:     [[RESHAPE_:%.+]] = tensor.reshape [[VAR_0_]]([[CST_]]) : (tensor<256xi32>, tensor<2xi64>) -> tensor<128x2xi32>
// CHECK:     [[EXTRACTED_SLICE_:%.+]] = tensor.extract_slice [[RESHAPE_]][0, 0] [128, 1] [1, 128]{{.*}} : tensor<128x2xi32> to tensor<128xi32>
// CHECK:     [[EXTRACTED_SLICE_0_:%.+]] = tensor.extract_slice [[RESHAPE_]][0, 1] [128, 1] [1, 128]{{.*}} : tensor<128x2xi32> to tensor<128xi32>
// CHECK:     [[REINTERPRET_CAST_1_:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xi32> to memref<128xi32, strided<[1]>>
// CHECK:     bufferization.materialize_in_destination [[EXTRACTED_SLICE_]] in writable [[REINTERPRET_CAST_1_]] : (tensor<128xi32>, memref<128xi32, strided<[1]>>) -> ()
// CHECK:     [[REINTERPRET_CAST_2_:%.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xi32> to memref<128xi32, strided<[1]>>
// CHECK:     bufferization.materialize_in_destination [[EXTRACTED_SLICE_0_]] in writable [[REINTERPRET_CAST_2_]] : (tensor<128xi32>, memref<128xi32, strided<[1]>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
