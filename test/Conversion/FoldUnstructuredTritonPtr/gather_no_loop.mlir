// RUN: triton-shared-opt --fold-unstructured-triton-ptr %s | FileCheck %s

module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %2 = arith.addi %1, %cst : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.load %4 : tensor<64x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %7, %5 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @gather_simple_no_loop([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<5> : tensor<64xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<10> : tensor<64xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:           [[VAR_1_:%.+]] = arith.divsi [[VAR_0_]], [[VAR_cst_0_]] : tensor<64xi32>
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[VAR_cst_]] : tensor<64xi32>
// CHECK:           [[VAR_3_:%.+]] = "tts.create_ptr"([[PARAM_0_]], [[VAR_2_]]) : (!tt.ptr<f32>, tensor<64xi32>) -> tensor<64x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_3_MEM_:%.+]] = tt.load [[VAR_3_]] : tensor<64x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tts.create_ptr"([[PARAM_1_]], [[VAR_0_]]) : (!tt.ptr<f32>, tensor<64xi32>) -> tensor<64x!tt.ptr<f32>>
// CHECK:           tt.store [[VAR_5_]], [[LOAD_VAR_3_MEM_]] : tensor<64x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
