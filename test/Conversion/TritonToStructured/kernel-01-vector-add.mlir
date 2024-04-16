// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @add_kernel_01234([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: i32) {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_1024_1_:%.+]] = arith.constant 1024 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_1024_1_]] : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_4_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[CST_1024_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_9_:%.+]] = arith.minsi [[VAR_7_]], [[VAR_8_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.subi [[VAR_9_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_11_:%.+]] = "tts.load"([[VAR_5_]], [[VAR_10_]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_3_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.addi [[VAR_13_]], [[CST_1024_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_16_:%.+]] = arith.minsi [[VAR_14_]], [[VAR_15_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.subi [[VAR_16_]], [[VAR_13_]] : index
// CHECK:           [[VAR_18_:%.+]] = "tts.load"([[VAR_12_]], [[VAR_17_]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.addf [[VAR_11_]], [[VAR_18_]] : tensor<1024xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[CST_1024_]] : index
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_24_:%.+]] = arith.minsi [[VAR_22_]], [[VAR_23_]] : index
// CHECK:           [[VAR_25_:%.+]] = arith.subi [[VAR_24_]], [[VAR_21_]] : index
// CHECK:           "tts.store"([[VAR_20_]], [[VAR_19_]], [[VAR_25_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
