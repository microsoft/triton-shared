// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @gather_simple_no_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_0 : tensor<64xi32>
      %5 = arith.addi %arg2, %c5_i32 : i32
      %6 = arith.remsi %5, %c64_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<64xi32>
      %8 = arith.addi %4, %7 : tensor<64xi32>
      %9 = tt.addptr %1, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
      %11 = tt.addptr %2, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %11, %10 : tensor<64x!tt.ptr<f32>>
      %12 = arith.addi %8, %cst : tensor<64xi32>
      %13 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %12, %13 : tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK:   tt.func public @gather_simple_no_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:     %cst = arith.constant dense<10> : tensor<64xi32>
// CHECK:     %c5_i32 = arith.constant 5 : i32
// CHECK:     %c64_i32 = arith.constant 64 : i32
// CHECK:     %cst_0 = arith.constant dense<64> : tensor<64xi32>
// CHECK:     %c2_i32 = arith.constant 2 : i32
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %1 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:     %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %2, %arg4 = %2) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:       %4 = arith.divsi %arg3, %cst : tensor<64xi32>
// CHECK:       %5 = arith.addi %arg2, %c5_i32 : i32
// CHECK:       %6 = arith.remsi %5, %c64_i32 : i32
// CHECK:       %7 = tt.splat %6 : i32 -> tensor<64xi32>
// CHECK:       %8 = arith.addi %4, %7 : tensor<64xi32>
// CHECK:       %cast = memref.cast %1 : memref<*xf32> to memref<?xf32>
// CHECK:       %9 = bufferization.to_tensor %cast restrict : memref<?xf32>
// CHECK:       %10 = tensor.empty() : tensor<64xf32>
// CHECK:       %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%8 : tensor<64xi32>) outs(%10 : tensor<64xf32>) {
// CHECK:       ^bb0(%in: i32, %out: f32):
// CHECK:         %14 = arith.index_cast %in : i32 to index
// CHECK:         %extracted = tensor.extract %9[%14] : tensor<?xf32>
// CHECK:         linalg.yield %extracted : f32
// CHECK:       } -> tensor<64xf32>
// CHECK:       %cast_1 = memref.cast %0 : memref<*xf32> to memref<?xf32>
// CHECK:       affine.for %arg5 = 0 to 64 {
// CHECK:         %extracted = tensor.extract %arg4[%arg5] : tensor<64xi32>
// CHECK:         %extracted_2 = tensor.extract %11[%arg5] : tensor<64xf32>
// CHECK:         %14 = arith.index_cast %extracted : i32 to index
// CHECK:         memref.store %extracted_2, %cast_1[%14] : memref<?xf32>
// CHECK:       }
// CHECK:       %12 = arith.addi %8, %cst_0 : tensor<64xi32>
// CHECK:       %13 = arith.addi %arg4, %cst_0 : tensor<64xi32>
// CHECK:       scf.yield %12, %13 : tensor<64xi32>, tensor<64xi32>
// CHECK:     }
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
