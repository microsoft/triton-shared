// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @addptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = tt.addptr %arg1, %c1_i32 : !tt.ptr<f32>, i32
    scf.for %arg2 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
      %2 = tt.addptr %0, %arg2 : !tt.ptr<f32>, i32
      %3 = tt.addptr %2, %c1_i32 : !tt.ptr<f32>, i32
      %4 = tt.addptr %1, %arg2 : !tt.ptr<f32>, i32
      %5 = tt.addptr %4, %c1_i32 : !tt.ptr<f32>, i32
      %6 = tt.load %2 : !tt.ptr<f32>
      %7 = tt.load %3 : !tt.ptr<f32>
      tt.store %4, %6 : !tt.ptr<f32>
      tt.store %5, %7 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @addptr(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c10_i32 = arith.constant 10 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK:     scf.for %arg8 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
// CHECK:       %0 = arith.index_cast %arg8 : i32 to index
// CHECK:       %1 = arith.addi %0, %c1 : index
// CHECK:       %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:       %2 = arith.addi %0, %c2 : index
// CHECK:       %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:       %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:       %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:       %3 = affine.load %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:       %4 = affine.load %reinterpret_cast_0[0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:       affine.store %3, %reinterpret_cast_1[0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:       affine.store %4, %reinterpret_cast_2[0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
