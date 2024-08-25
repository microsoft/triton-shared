// RUN: triton-shared-opt --linalg-to-linear-algebra-subprograms %s | FileCheck %s

module {
  func.func @bare_matmul(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg9, %c128_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %arg10, %c128_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = arith.muli %1, %4 : index
    %6 = arith.addi %5, %3 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [128, 128], strides: [%4, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<128x128xf32>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
    %8 = arith.index_cast %arg4 : i32 to index
    %9 = arith.muli %1, %8 : index
    %10 = arith.addi %9, %3 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [128, 128], strides: [%8, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<128x128xf32>
    %11 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x128xf32>
    %12 = tensor.empty() : tensor<128x128xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %14 = linalg.matmul ins(%7, %11 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%13 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [128, 128], strides: [%8, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %14 in writable %reinterpret_cast_2 : (tensor<128x128xf32>, memref<128x128xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

// CHECK: module {
// CHECK:   func.func private @cblas_sgemm(i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32)
// CHECK:   func.func @bare_matmul(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
// CHECK:     [[C128_I32:%.+]] = arith.constant 128 : i32
// CHECK:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:     [[VAR_0:%.+]] = arith.muli %arg9, [[C128_I32]] : i32
// CHECK:     [[VAR_1:%.+]] = arith.index_cast [[VAR_0]] : i32 to index
// CHECK:     [[VAR_2:%.+]] = arith.muli %arg10, [[C128_I32]] : i32
// CHECK:     [[VAR_3:%.+]] = arith.index_cast [[VAR_2]] : i32 to index
// CHECK:     [[VAR_4:%.+]] = arith.index_cast %arg5 : i32 to index
// CHECK:     [[VAR_5:%.+]] = arith.muli [[VAR_1]], [[VAR_4]] : index
// CHECK:     [[VAR_6:%.+]] = arith.addi [[VAR_5]], [[VAR_3]] : index
// CHECK:     [[REINTERPRET_CAST:%.+]] = memref.reinterpret_cast %arg0 to offset: [[VAR_6]]{{.*}} : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
// CHECK:     [[ALLOC:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:     memref.copy [[REINTERPRET_CAST]], [[ALLOC]]{{.*}} : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<128x128xf32>
// CHECK:     [[VAR_7:%.+]] = bufferization.to_tensor [[ALLOC]] restrict writable{{.*}} : memref<128x128xf32>
// CHECK:     [[VAR_8:%.+]] = arith.index_cast %arg4 : i32 to index
// CHECK:     [[VAR_9:%.+]] = arith.muli [[VAR_1]], [[VAR_8]] : index
// CHECK:     [[VAR_10:%.+]] = arith.addi [[VAR_9]], [[VAR_3]] : index
// CHECK:     [[REINTERPRET_CAST_0:%.+]] = memref.reinterpret_cast %arg1 to offset: [[VAR_10]]{{.*}} : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
// CHECK:     [[ALLOC_1:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:     memref.copy [[REINTERPRET_CAST_0]], [[ALLOC_1]]{{.*}} : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<128x128xf32>
// CHECK:     [[VAR_11:%.+]] = bufferization.to_tensor [[ALLOC_1]] restrict writable{{.*}} : memref<128x128xf32>
// CHECK:     [[VAR_12:%.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK:     [[VAR_13:%.+]] = linalg.fill ins([[CST]] : f32) outs([[VAR_12]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:     [[VAR_14:%.+]] = linalg.matmul ins([[VAR_7]], [[VAR_11]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[VAR_13]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:     [[REINTERPRET_CAST_2:%.+]] = memref.reinterpret_cast %arg2 to offset: [[VAR_10]]{{.*}} : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
// CHECK:     bufferization.materialize_in_destination [[VAR_14]] in writable [[REINTERPRET_CAST_2]] : (tensor<128x128xf32>, memref<128x128xf32, strided<[?, 1], offset: ?>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   %cst_5 = arith.constant 1.000000e+00 : f32
// CHECK:   call @cblas_sgemm{{.*}} : (i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32) -> ()
// CHECK: }