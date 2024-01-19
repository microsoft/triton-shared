// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

module {
  tt.func public @softmax_kernel_012345(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : (i32) -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.splat %cst : (f32) -> tensor<128xf32>
    %9 = tt.load %5, %7, %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32>
    %10 = "tt.reduce"(%9) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.cmpf ogt, %arg5, %arg6 : f32
      %22 = arith.select %21, %arg5, %arg6 : f32
      tt.reduce.return %22 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %11 = tt.splat %10 : (f32) -> tensor<128xf32>
    %12 = arith.subf %9, %11 : tensor<128xf32>
    %13 = math.exp %12 : tensor<128xf32>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %15 = tt.splat %14 : (f32) -> tensor<128xf32>
    %16 = arith.divf %13, %15 : tensor<128xf32>
    %17 = arith.muli %0, %arg3 : i32
    %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
    %19 = tt.splat %18 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
    %20 = tt.addptr %19, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %20, %16, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32>
    tt.return
  }
}

// CHECK:         tt.func public @softmax_kernel_012345([[PARAM_0_:%.+]]: !tt.ptr<f32, 1>, [[PARAM_1_:%.+]]: !tt.ptr<f32, 1>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, parent_sizes: [0] : <f32, 1> to tensor<128x!tt.ptr<f32, 1>>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[CST_128_]] : index
// CHECK:           [[VAR_6_:%.+]] = "tts.load"([[VAR_3_]], [[VAR_5_]], [[CST_0_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32, 1>>, index, f32) -> tensor<128xf32>
// CHECK:           [[VAR_7_:%.+]] = "tt.reduce"([[VAR_6_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[arg5_:%.+]]: f32, [[arg6_:%.+]]: f32):
// CHECK:             [[VAR_19_:%.+]] = arith.cmpf ogt, [[arg5_]], [[arg6_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[arg5_]], [[arg6_]] : f32
// CHECK:             tt.reduce.return [[VAR_20_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_8_:%.+]] = tt.splat [[VAR_7_]] : (f32) -> tensor<128xf32>
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[VAR_6_]], [[VAR_8_]] : tensor<128xf32>
// CHECK:           [[VAR_10_:%.+]] = math.exp [[VAR_9_]] : tensor<128xf32>
// CHECK:           [[VAR_11_:%.+]] = "tt.reduce"([[VAR_10_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[arg5_]]: f32, [[arg6_]]: f32):
// CHECK:             [[VAR_19_1_:%.+]] = arith.addf [[arg5_]], [[arg6_]] : f32
// CHECK:             tt.reduce.return [[VAR_19_1_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_12_:%.+]] = tt.splat [[VAR_11_]] : (f32) -> tensor<128xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.divf [[VAR_10_]], [[VAR_12_]] : tensor<128xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i32 to index
// CHECK-DAG:       [[VAR_16_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128], strides: [1], offsets: {{.}}[[VAR_15_]]{{.}}, parent_sizes: [0] : <f32, 1> to tensor<128x!tt.ptr<f32, 1>>
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_18_:%.+]] = arith.minsi [[VAR_17_]], [[CST_128_]] : index
// CHECK:           "tts.store"([[VAR_16_]], [[VAR_13_]], [[VAR_18_]]) <{static_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32, 1>>, tensor<128xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
