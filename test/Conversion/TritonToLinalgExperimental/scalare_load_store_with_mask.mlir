// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure scalar load store with mask generate if correctly.
// CHECK-LABEL: masked_scalar_gather_scatter
// CHECK-SAME:(%[[arg0:.*]]: memref<*xi32> {tt.divisibility = 16 : i32}, %[[arg1:.*]]: memref<*xi32> {tt.divisibility = 16 : i32}, %[[arg2:.*]]: i1, %[[arg3:.*]]: i1
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[c99_i32:.*]] = arith.constant 99 : i32
// CHECK: %[[if:.*]] = scf.if %[[arg2]] -> (i32) {
// CHECK:   %reinterpret_cast = memref.reinterpret_cast %[[arg0]] to offset: [%[[c0]]], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:   %[[ld:.*]] = affine.load %reinterpret_cast[0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:   scf.yield %[[ld]] : i32
// CHECK: } else {
// CHECK:  scf.yield %[[c99_i32]] : i32
// CHECK: }
// CHECK: scf.if %[[arg3]] {
// CHECK:   %reinterpret_cast = memref.reinterpret_cast %[[arg1]] to offset: [%[[c0]]], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:   affine.store %[[if]], %reinterpret_cast[0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK: }

module {
  tt.func public @masked_scalar_gather_scatter(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i1, %arg3: i1) attributes {noinline = false} {
    %c99_i32 = arith.constant 99 : i32
    %0 = tt.load %arg0, %arg2, %c99_i32 : !tt.ptr<i32>
    tt.store %arg1, %0, %arg3 : !tt.ptr<i32>
    tt.return
  }
}
