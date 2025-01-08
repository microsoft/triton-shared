// RUN: triton-shared-opt --fold-unstructured-ptr --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:         tt.func public @gather_simple_mask_no_other([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<4> : tensor<64xi32>
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : i32
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<64> : tensor<64xi32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]]:3 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[CST_8_]], [[VAR_arg4_:%.+]] = [[VAR_2_]], [[VAR_arg5_:%.+]] = [[VAR_2_]]) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.divsi [[VAR_arg4_]], [[VAR_cst_0_]] : tensor<64xi32>
// CHECK-DAG:         [[VAR_5_:%.+]] = tt.splat [[VAR_arg3_]] : i32 -> tensor<64xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[VAR_5_]] : tensor<64xi32>
// CHECK-DAG:         [[VAR_cast_:%.+]] = memref.cast [[VAR_1_]] : memref<*xf32> to memref<?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = bufferization.to_tensor [[VAR_cast_]] restrict : memref<?xf32>
// CHECK-DAG:         [[VAR_8_:%.+]] = tensor.empty() : tensor<64xf32>
// CHECK:             [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_6_]] : tensor<64xi32>, tensor<64xi1>) outs([[VAR_8_]] : tensor<64xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: i1, [[IN_2_:%.+]]: f32):
// CHECK-DAG:           [[VAR_13_:%.+]] = scf.if [[IN_1_]] -> (f32) {
// CHECK-DAG:             [[VAR_14_:%.+]] = arith.index_cast [[IN_0_]] : i32 to index
// CHECK:                 [[VAR_extracted_:%.+]] = tensor.extract [[VAR_7_]]{{.}}[[VAR_14_]]{{.}} : tensor<?xf32>
// CHECK:                 scf.yield [[VAR_extracted_]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield [[CST_0_dot_000000_]] : f32
// CHECK:               }
// CHECK:               linalg.yield [[VAR_13_]] : f32
// CHECK:             } -> tensor<64xf32>
// CHECK:             [[VAR_cast_2_:%.+]] = memref.cast [[VAR_0_]] : memref<*xf32> to memref<?xf32>
// CHECK:             affine.for [[I_0_:%.+]] = 0 to 64 {
// CHECK-DAG:           [[VAR_extracted_1_:%.+]] = tensor.extract [[VAR_arg5_]]{{.}}[[I_0_]]{{.}} : tensor<64xi32>
// CHECK-DAG:           [[VAR_extracted_3_:%.+]] = tensor.extract [[VAR_9_]]{{.}}[[I_0_]]{{.}} : tensor<64xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.index_cast [[VAR_extracted_1_]] : i32 to index
// CHECK:               memref.store [[VAR_extracted_3_]], [[VAR_cast_2_]]{{.}}[[VAR_13_1_]]{{.}} : memref<?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_arg3_]], [[CST_16_]] : i32
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_cst_1_]] : tensor<64xi32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_arg5_]], [[VAR_cst_1_]] : tensor<64xi32>
// CHECK:             scf.yield [[VAR_10_]], [[VAR_11_]], [[VAR_12_]] : i32, tensor<64xi32>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
