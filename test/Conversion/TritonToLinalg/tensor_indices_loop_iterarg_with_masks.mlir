// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// IR from python/examples/test_tensor_index_iterargs.py
module {
  tt.func public @addptr_with_masks(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<-1.100000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %4:2 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %0, %arg5 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %5 = arith.cmpi slt, %arg4, %1 : tensor<4xi32>
      %6 = tt.addptr %2, %arg4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %7 = tt.load %6, %5, %cst : tensor<4x!tt.ptr<f32>>
      %8 = tt.addptr %3, %arg5 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %8, %7 : tensor<4x!tt.ptr<f32>>
      %9 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      %10 = arith.addi %arg5, %cst_0 : tensor<4xi32>
      scf.yield %9, %10 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @addptr_with_masks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_minus_1_dot_100000_:%.+]] = arith.constant -1.100000e+01 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_4_]] : i32) outs([[VAR_0_]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<4xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_4_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i32
// CHECK:             linalg.yield [[VAR_5_]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]]:4 = scf.for [[VAR_arg9_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] iter_args([[VAR_arg10_:%.+]] = [[VAR_2_]], [[VAR_arg11_:%.+]] = [[CST_0_1_]], [[VAR_arg12_:%.+]] = [[VAR_2_]], [[VAR_arg13_:%.+]] = [[CST_0_1_]]) -> (tensor<4xi32>, index, tensor<4xi32>, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg11_]]{{.}}, sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}} : memref<*xf32> to memref<4xf32, strided<[?], offset: ?>>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = arith.addi [[VAR_arg11_]], [[CST_4_1_]] : index
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:             [[VAR_6_:%.+]] = arith.minsi [[VAR_4_1_]], [[VAR_5_1_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.subi [[VAR_6_]], [[VAR_arg11_]] : index
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_4_1_]] : index
// CHECK:             scf.if [[VAR_8_]] {
// CHECK:               linalg.fill ins([[CST_minus_1_dot_100000_]] : f32) outs([[RES_]] : memref<4xf32>)
// CHECK:             }
// CHECK-DAG:         [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_7_]]{{.}} [1] : memref<4xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
// CHECK-DAG:         [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_7_]]{{.}} [1] : memref<4xf32> to memref<?xf32, strided<[1]>>
// CHECK:             memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4xf32>
// CHECK-DAG:         [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg13_]]{{.}}, sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}} : memref<*xf32> to memref<4xf32, strided<[?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<4xf32>, memref<4xf32, strided<[?], offset: ?>>) -> ()
// CHECK:             [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg10_]], [[VAR_1_]] : tensor<4xi32>, tensor<4xi32>) outs([[VAR_arg10_]] : tensor<4xi32>) {
// CHECK:             ^bb0([[IN_1_:%.+]]: i32, [[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: i32):
// CHECK:               [[VAR_13_:%.+]] = arith.addi [[IN_1_]], [[IN_2_]] : i32
// CHECK:               linalg.yield [[VAR_13_]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg12_]], [[VAR_1_]] : tensor<4xi32>, tensor<4xi32>) outs([[VAR_arg12_]] : tensor<4xi32>) {
// CHECK:             ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: i32, [[IN_6_:%.+]]: i32):
// CHECK:               [[VAR_13_1_:%.+]] = arith.addi [[IN_4_]], [[IN_5_]] : i32
// CHECK:               linalg.yield [[VAR_13_1_]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             [[VAR_12_:%.+]] = arith.addi [[VAR_arg13_]], [[CST_4_1_]] : index
// CHECK:             scf.yield [[VAR_10_]], [[VAR_4_1_]], [[VAR_11_]], [[VAR_12_]] : tensor<4xi32>, index, tensor<4xi32>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
