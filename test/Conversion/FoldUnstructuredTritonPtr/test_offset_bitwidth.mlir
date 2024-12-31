// RUN: triton-shared-opt --fold-unstructured-triton-ptr="offset-bit-width=64" %s | FileCheck %s

module {
  tt.func public @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.get_program_id x : i32
    %2 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.extsi %2 : tensor<64xi32> to tensor<64xi64>
    %5 = tt.splat %3 : i64 -> tensor<64xi64>
    %6 = arith.addi %4, %5 : tensor<64xi64>
    %7 = tt.splat %1 : i32 -> tensor<64xi32>
    %8 = arith.addi %2, %7 : tensor<64xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %10 = tt.addptr %9, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi64>
    %11 = tt.addptr %10, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %13:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %11, %arg4 = %0) -> (tensor<64x!tt.ptr<f32>>, tensor<64xi32>)  : i32 {
      %14 = tt.load %arg3 : tensor<64x!tt.ptr<f32>>
      %15 = tt.addptr %12, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %15, %14 : tensor<64x!tt.ptr<f32>>
      %16 = tt.addptr %arg3, %7 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %17 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %16, %17 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @gather_simple([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<64> : tensor<64xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<10> : tensor<64xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.get_program_id x : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.divsi [[VAR_0_]], [[VAR_cst_0_]] : tensor<64xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.extsi [[VAR_1_]] : i32 to i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.extsi [[VAR_2_]] : tensor<64xi32> to tensor<64xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = tt.splat [[VAR_3_]] : i64 -> tensor<64xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[VAR_5_]] : tensor<64xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = tt.splat [[VAR_1_]] : i32 -> tensor<64xi32>
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_2_]], [[VAR_7_]] : tensor<64xi32>
// CHECK:           [[VAR_9_:%.+]] = arith.extsi [[VAR_8_]] : tensor<64xi32> to tensor<64xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.addi [[VAR_6_]], [[VAR_9_]] : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_0_]]) -> (tensor<64xi32>)  : i32 {
// CHECK:             [[VAR_12_:%.+]] = "tts.create_ptr"([[PARAM_0_]], [[VAR_10_]]) : (!tt.ptr<f32>, tensor<64xi64>) -> tensor<64x!tt.ptr<f32>>
// CHECK-DAG:         [[LOAD_VAR_12_MEM_:%.+]] = tt.load [[VAR_12_]] : tensor<64x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.extsi [[VAR_arg3_]] : tensor<64xi32> to tensor<64xi64>
// CHECK:             [[VAR_15_:%.+]] = "tts.create_ptr"([[PARAM_1_]], [[VAR_14_]]) : (!tt.ptr<f32>, tensor<64xi64>) -> tensor<64x!tt.ptr<f32>>
// CHECK:             tt.store [[VAR_15_]], [[LOAD_VAR_12_MEM_]] : tensor<64x!tt.ptr<f32>>
// CHECK:             [[VAR_16_:%.+]] = arith.addi [[VAR_arg3_]], [[VAR_cst_]] : tensor<64xi32>
// CHECK:             scf.yield [[VAR_16_]] : tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
