// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<3> : tensor<2x2xi32>
    %cst_0 = arith.constant dense<1> : tensor<2x2xi32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg2, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.addptr %arg5, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %20 = tt.addptr %arg6, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %19, %arg9 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.addptr %arg9, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %27, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %22 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %23 = tt.addptr %22, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %25 = tt.addptr %24, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
  tt.func public @nested2_use_loop_results(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c4_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg6, %19 : tensor<2x2x!tt.ptr<f32>>
      %20 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %20, %arg9 = %21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %23 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %23 : tensor<2x2x!tt.ptr<f32>>
        %24 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %25 = tt.addptr %arg9, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %24, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      scf.yield %22#0, %22#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
  tt.func public @nested3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %21 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %24 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %25 = tt.load %24 : tensor<2x2x!tt.ptr<f32>>
        %26:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %24, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %27 = tt.addptr %arg11, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg12, %21 : tensor<2x2x!tt.ptr<f32>>
          %29 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %29, %25 : tensor<2x2x!tt.ptr<f32>>
          %30 = tt.addptr %29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
          %31 = tt.addptr %30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          scf.yield %27, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        scf.yield %26#0, %26#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %23 = tt.addptr %22#0, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %22#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @nested2_complex_body
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[PARAM_2_]], [[CST_2_]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[CST_0_1_]], [[VAR_arg12_:%.+]] = [[CST_0_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg11_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg12_]], [[CST_1_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = scf.for [[VAR_arg13_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg14_:%.+]] = [[VAR_5_]], [[VAR_arg15_:%.+]] = [[VAR_6_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg15_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg14_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[RES_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:               [[VAR_12_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<2x2xf32>
// CHECK:               bufferization.materialize_in_destination [[VAR_12_]] in writable [[VAR_reinterpret_cast_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.addi [[VAR_arg14_]], [[CST_3_]] : index
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_arg15_]], [[CST_3_]] : index
// CHECK:               scf.yield [[VAR_13_]], [[VAR_14_]] : index, index
// CHECK:             }
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_arg11_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_3_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[CST_1_1_]] : index
// CHECK:             scf.yield [[VAR_9_]], [[VAR_11_]] : index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
//
// CHECK-LABEL:  func.func @nested2_use_loop_results
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[PARAM_3_]], [[CST_4_]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[CST_0_1_]], [[VAR_arg12_:%.+]] = [[CST_0_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg12_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg11_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:             [[VAR_5_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<2x2xf32>
// CHECK:             bufferization.materialize_in_destination [[VAR_5_]] in writable [[VAR_reinterpret_cast_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg11_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]]:2 = scf.for [[VAR_arg13_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg14_:%.+]] = [[VAR_6_]], [[VAR_arg15_:%.+]] = [[VAR_7_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg15_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg14_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[RES_1_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_2_]], [[RES_1_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:               [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<2x2xf32>
// CHECK:               bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.addi [[VAR_arg14_]], [[VAR_3_]] : index
// CHECK-DAG:           [[VAR_11_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_3_]] : index
// CHECK:               scf.yield [[VAR_10_]], [[VAR_11_]] : index, index
// CHECK:             }
// CHECK:             scf.yield [[VAR_8_]]#0, [[VAR_8_]]#1 : index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
//
// CHECK-LABEL:  func.func @nested3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:3 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[CST_0_1_]], [[VAR_arg12_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg13_:%.+]] = [[CST_0_1_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg11_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:         [[VAR_5_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:         [[VAR_6_:%.+]]:3 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_arg11_]], [[VAR_arg16_:%.+]] = [[VAR_arg12_]], [[VAR_arg17_:%.+]] = [[VAR_arg13_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_8_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[RES_1_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_1_]], [[RES_1_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:           [[VAR_10_:%.+]]:3 = scf.for [[VAR_arg18_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg19_:%.+]] = [[VAR_8_]], [[VAR_arg20_:%.+]] = [[VAR_arg16_]], [[VAR_arg21_:%.+]] = [[VAR_arg17_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:             [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg21_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[VAR_11_:%.+]] = arith.addi [[VAR_arg19_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_11_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                 memref.copy [[VAR_reinterpret_cast_4_]], [[RES_2_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                 [[VAR_12_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<2x2xf32>
// CHECK:                 bufferization.materialize_in_destination [[VAR_5_]] in writable [[VAR_reinterpret_cast_3_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_13_:%.+]] = arith.addi [[VAR_arg21_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_13_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                 bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_6_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_14_:%.+]] = arith.addi [[VAR_13_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_14_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                 bufferization.materialize_in_destination [[VAR_12_]] in writable [[VAR_reinterpret_cast_7_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_15_:%.+]] = arith.addi [[VAR_14_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_8_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_15_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                 scf.yield [[VAR_11_]], [[VAR_reinterpret_cast_8_]], [[VAR_15_]] : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:               }
// CHECK:               scf.yield [[VAR_10_]]#0, [[VAR_10_]]#1, [[VAR_10_]]#2 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:             }
// CHECK:             [[VAR_7_:%.+]] = arith.addi [[VAR_6_]]#0, [[VAR_3_]] : index
// CHECK:             scf.yield [[VAR_7_]], [[VAR_6_]]#1, [[VAR_6_]]#2 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
