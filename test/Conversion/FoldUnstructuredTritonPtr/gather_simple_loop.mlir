// RUN: triton-shared-opt --fold-unstructured-triton-ptr %s | FileCheck %s

module {
  tt.func public @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
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

// CHECK:         tt.func public @gather_simple([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<64> : tensor<64xi32>
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<10> : tensor<64xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_0_]], [[VAR_arg4_:%.+]] = [[VAR_0_]]) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.divsi [[VAR_arg3_]], [[VAR_cst_0_]] : tensor<64xi32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_5_]] : i32
// CHECK:             [[VAR_4_:%.+]] = arith.remsi [[VAR_3_]], [[CST_64_]] : i32
// CHECK:             [[VAR_5_:%.+]] = tt.splat [[VAR_4_]] : i32 -> tensor<64xi32>
// CHECK:             [[VAR_6_:%.+]] = arith.addi [[VAR_2_]], [[VAR_5_]] : tensor<64xi32>
// CHECK:             [[VAR_7_:%.+]] = "tts.make_unstructured_tptr"([[PARAM_0_]], [[VAR_6_]]) : (!tt.ptr<f32>, tensor<64xi32>) -> tensor<64x!tt.ptr<f32>>
// CHECK-DAG:         [[LOAD_VAR_7_MEM_:%.+]] = tt.load [[VAR_7_]] : tensor<64x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_9_:%.+]] = "tts.make_unstructured_tptr"([[PARAM_1_]], [[VAR_arg4_]]) : (!tt.ptr<f32>, tensor<64xi32>) -> tensor<64x!tt.ptr<f32>>
// CHECK:             tt.store [[VAR_9_]], [[LOAD_VAR_7_MEM_]] : tensor<64x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_6_]], [[VAR_cst_]] : tensor<64xi32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_cst_]] : tensor<64xi32>
// CHECK:             scf.yield [[VAR_10_]], [[VAR_11_]] : tensor<64xi32>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
