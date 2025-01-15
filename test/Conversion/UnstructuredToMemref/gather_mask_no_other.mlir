// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @gather_simple_mask_no_other(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant dense<4> : tensor<64xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c8_i32, %arg4 = %0, %arg5 = %0) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg4, %cst_0 : tensor<64xi32>
      %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
      %7 = tt.addptr %1, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %8 = tt.load %7, %6 : tensor<64x!tt.ptr<f32>>
      %9 = tt.addptr %2, %arg5 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %9, %8 : tensor<64x!tt.ptr<f32>>
      %10 = arith.addi %arg3, %c16_i32 : i32
      %11 = arith.addi %arg4, %cst : tensor<64xi32>
      %12 = arith.addi %arg5, %cst : tensor<64xi32>
      scf.yield %10, %11, %12 : i32, tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK:   tt.func public @gather_simple_mask_no_other(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %c8_i32 = arith.constant 8 : i32
// CHECK:     %cst_0 = arith.constant dense<4> : tensor<64xi32>
// CHECK:     %c16_i32 = arith.constant 16 : i32
// CHECK:     %cst_1 = arith.constant dense<64> : tensor<64xi32>
// CHECK:     %c2_i32 = arith.constant 2 : i32
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %1 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
// CHECK:     %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:     %3:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c8_i32, %arg4 = %2, %arg5 = %2) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:       %4 = arith.divsi %arg4, %cst_0 : tensor<64xi32>
// CHECK:       %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
// CHECK:       %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
// CHECK:       %cast = memref.cast %1 : memref<*xf32> to memref<?xf32>
// CHECK:       %7 = bufferization.to_tensor %cast restrict : memref<?xf32>
// CHECK:       %8 = tensor.empty() : tensor<64xf32>
// CHECK:       %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %6 : tensor<64xi32>, tensor<64xi1>) outs(%8 : tensor<64xf32>) {
// CHECK:       ^bb0(%in: i32, %in_3: i1, %out: f32):
// CHECK:         %13 = scf.if %in_3 -> (f32) {
// CHECK:           %14 = arith.index_cast %in : i32 to index
// CHECK:           %extracted = tensor.extract %7[%14] : tensor<?xf32>
// CHECK:           scf.yield %extracted : f32
// CHECK:         } else {
// CHECK:           scf.yield %cst : f32
// CHECK:         }
// CHECK:         linalg.yield %13 : f32
// CHECK:       } -> tensor<64xf32>
// CHECK:       %cast_2 = memref.cast %0 : memref<*xf32> to memref<?xf32>
// CHECK:       affine.for %arg6 = 0 to 64 {
// CHECK:         %extracted = tensor.extract %arg5[%arg6] : tensor<64xi32>
// CHECK:         %extracted_3 = tensor.extract %9[%arg6] : tensor<64xf32>
// CHECK:         %13 = arith.index_cast %extracted : i32 to index
// CHECK:         memref.store %extracted_3, %cast_2[%13] : memref<?xf32>
// CHECK:       }
// CHECK:       %10 = arith.addi %arg3, %c16_i32 : i32
// CHECK:       %11 = arith.addi %arg4, %cst_1 : tensor<64xi32>
// CHECK:       %12 = arith.addi %arg5, %cst_1 : tensor<64xi32>
// CHECK:       scf.yield %10, %11, %12 : i32, tensor<64xi32>, tensor<64xi32>
// CHECK:     }
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
