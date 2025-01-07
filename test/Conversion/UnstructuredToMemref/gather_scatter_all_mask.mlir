// RUN: triton-shared-opt --fold-unstructured-triton-ptr --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<9.900000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %cst_1 = arith.constant dense<64> : tensor<4xi32>
    %cst_2 = arith.constant dense<3> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_2 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
      %8 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %9 = tt.load %8, %7, %cst : tensor<4x!tt.ptr<f32>>
      %10 = tt.addptr %2, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %10, %9, %7 : tensor<4x!tt.ptr<f32>>
      %11 = arith.addi %6, %cst_0 : tensor<4xi32>
      %12 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      scf.yield %11, %12 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK:   tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:     %cst = arith.constant 9.900000e+01 : f32
// CHECK:     %cst_0 = arith.constant dense<3> : tensor<4xi32>
// CHECK:     %cst_1 = arith.constant dense<64> : tensor<4xi32>
// CHECK:     %cst_2 = arith.constant dense<4> : tensor<4xi32>
// CHECK:     %c2_i32 = arith.constant 2 : i32
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %1 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:     %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %2, %arg4 = %2) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:       %4 = arith.divsi %arg3, %cst_0 : tensor<4xi32>
// CHECK:       %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
// CHECK:       %6 = arith.addi %4, %5 : tensor<4xi32>
// CHECK:       %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
// CHECK:       %cast = memref.cast %1 : memref<*xf32> to memref<?xf32>
// CHECK:       %8 = bufferization.to_tensor %cast restrict : memref<?xf32>
// CHECK:       %9 = tensor.empty() : tensor<4xf32>
// CHECK:       %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %7 : tensor<4xi32>, tensor<4xi1>) outs(%9 : tensor<4xf32>) {
// CHECK:       ^bb0(%in: i32, %in_4: i1, %out: f32):
// CHECK:         %13 = scf.if %in_4 -> (f32) {
// CHECK:           %14 = arith.index_cast %in : i32 to index
// CHECK:           %extracted = tensor.extract %8[%14] : tensor<?xf32>
// CHECK:           scf.yield %extracted : f32
// CHECK:         } else {
// CHECK:           scf.yield %cst : f32
// CHECK:         }
// CHECK:         linalg.yield %13 : f32
// CHECK:       } -> tensor<4xf32>
// CHECK:       %cast_3 = memref.cast %0 : memref<*xf32> to memref<?xf32>
// CHECK:       affine.for %arg5 = 0 to 4 {
// CHECK:         %extracted = tensor.extract %7[%arg5] : tensor<4xi1>
// CHECK:         scf.if %extracted {
// CHECK:           %extracted_4 = tensor.extract %6[%arg5] : tensor<4xi32>
// CHECK:           %extracted_5 = tensor.extract %10[%arg5] : tensor<4xf32>
// CHECK:           %13 = arith.index_cast %extracted_4 : i32 to index
// CHECK:           memref.store %extracted_5, %cast_3[%13] : memref<?xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       %11 = arith.addi %6, %cst_2 : tensor<4xi32>
// CHECK:       %12 = arith.addi %arg4, %cst_2 : tensor<4xi32>
// CHECK:       scf.yield %11, %12 : tensor<4xi32>, tensor<4xi32>
// CHECK:     }
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
