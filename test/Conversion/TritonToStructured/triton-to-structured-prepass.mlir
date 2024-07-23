// RUN: triton-shared-opt --triton-to-structured="run-prepass-only=true" --split-input-file %s | FileCheck %s

module {
  tt.func public @nested1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg4 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg5 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg6 : i32 -> tensor<2x1xi32>
    %13 = arith.muli %12, %1 : tensor<2x1xi32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %16 = tt.splat %arg7 : i32 -> tensor<1x2xi32>
    %17 = arith.muli %16, %4 : tensor<1x2xi32>
    %18 = tt.broadcast %15 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %19 = tt.broadcast %17 : tensor<1x2xi32> -> tensor<2x2xi32>
    %20 = tt.addptr %18, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %21 = arith.muli %arg5, %c32_i32 : i32
    %22 = tt.splat %21 : i32 -> tensor<2x2xi32>
    %23:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %11, %arg10 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %24 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %24 : tensor<2x2x!tt.ptr<f32>>
      %25 = tt.addptr %arg9, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %26 = tt.addptr %arg10, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %25, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:           tt.func public @nested1([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32, [[arg4_:.+]]: i32, [[arg5_:.+]]: i32, [[arg6_:.+]]: i32, [[arg7_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:         [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:         [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:         [[VAR_0_:%.+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_1_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG:         [[VAR_2_:%.+]] = tt.splat [[arg4_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_2_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_4_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK-DAG:         [[VAR_5_:%.+]] = tt.splat [[arg5_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.muli [[VAR_4_]], [[VAR_5_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_7_:%.+]] = tt.broadcast [[VAR_3_]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:             [[VAR_8_:%.+]] = tt.broadcast [[VAR_6_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_7_]], [[VAR_8_]] : tensor<2x2xi32>
// CHECK-DAG:         [[VAR_10_:%.+]] = tt.splat [[arg0_]] : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = tt.addptr [[VAR_10_]], [[VAR_9_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_12_:%.+]] = tt.splat [[arg6_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.muli [[VAR_12_]], [[VAR_1_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_14_:%.+]] = tt.splat [[arg1_]] : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_15_:%.+]] = tt.addptr [[VAR_14_]], [[VAR_13_]] : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
// CHECK-DAG:         [[VAR_16_:%.+]] = tt.splat [[arg7_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[VAR_4_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_18_:%.+]] = tt.broadcast [[VAR_15_]] : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
// CHECK:             [[VAR_19_:%.+]] = tt.broadcast [[VAR_17_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_20_:%.+]] = tt.addptr [[VAR_18_]], [[VAR_19_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.muli [[arg5_]], [[CST_32_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = tt.splat [[VAR_21_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_23_:%.+]]:5 = "tts.get_structured_state"([[VAR_11_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:         [[VAR_24_:%.+]]:5 = "tts.get_structured_state"([[VAR_20_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]]:10 = scf.for [[VAR_arg8_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg9_:%.+]] = [[VAR_23_]]#0, [[VAR_arg10_:%.+]] = [[VAR_23_]]#1, [[VAR_arg11_:%.+]] = [[VAR_23_]]#2, [[VAR_arg12_:%.+]] = [[VAR_23_]]#3, [[VAR_arg13_:%.+]] = [[VAR_23_]]#4, [[VAR_arg14_:%.+]] = [[VAR_24_]]#0, [[VAR_arg15_:%.+]] = [[VAR_24_]]#1, [[VAR_arg16_:%.+]] = [[VAR_24_]]#2, [[VAR_arg17_:%.+]] = [[VAR_24_]]#3, [[VAR_arg18_:%.+]] = [[VAR_24_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:           [[LOAD_VAR_arg9_MEM_:%.+]] = tt.load [[VAR_arg9_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:               tt.store [[VAR_arg14_]], [[LOAD_VAR_arg9_MEM_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_27_:%.+]] = tt.addptr [[VAR_arg9_]], [[VAR_22_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:           [[VAR_28_:%.+]] = tt.addptr [[VAR_arg14_]], [[VAR_22_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]]:5 = "tts.get_structured_state"([[VAR_27_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:           [[VAR_30_:%.+]]:5 = "tts.get_structured_state"([[VAR_28_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:               scf.yield [[VAR_29_]]#0, [[VAR_29_]]#1, [[VAR_29_]]#2, [[VAR_29_]]#3, [[VAR_29_]]#4, [[VAR_30_]]#0, [[VAR_30_]]#1, [[VAR_30_]]#2, [[VAR_30_]]#3, [[VAR_30_]]#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:             }
// CHECK:             tt.return
// CHECK:           }
// CHECK:         }

// ----

module {
  tt.func public @nested2(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg4 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg5 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg6 : i32 -> tensor<2x1xi32>
    %13 = arith.muli %12, %1 : tensor<2x1xi32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %16 = tt.splat %arg7 : i32 -> tensor<1x2xi32>
    %17 = arith.muli %16, %4 : tensor<1x2xi32>
    %18 = tt.broadcast %15 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %19 = tt.broadcast %17 : tensor<1x2xi32> -> tensor<2x2xi32>
    %20 = tt.addptr %18, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %21 = arith.muli %arg5, %c32_i32 : i32
    %22 = tt.splat %21 : i32 -> tensor<2x2xi32>
    %23:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %11, %arg10 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %24 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %24 : tensor<2x2x!tt.ptr<f32>>
      %25 = tt.addptr %arg9, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %26 = tt.addptr %arg10, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %27:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %25, %arg13 = %26) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %28 = tt.load %arg12 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg13, %28 : tensor<2x2x!tt.ptr<f32>>
        %29 = tt.addptr %arg12, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.addptr %arg13, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %29, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      scf.yield %27#0, %27#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:           tt.func public @nested2([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32, [[arg4_:.+]]: i32, [[arg5_:.+]]: i32, [[arg6_:.+]]: i32, [[arg7_]]: i32) attributes {noinline = false} {
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:         [[CST_2_1_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:         [[CST_32_1_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:         [[VAR_0_1_:%.+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_1_1_:%.+]] = tt.expand_dims [[VAR_0_1_]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG:         [[VAR_2_1_:%.+]] = tt.splat [[arg4_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_1_:%.+]] = arith.muli [[VAR_1_1_]], [[VAR_2_1_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = tt.expand_dims [[VAR_0_1_]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = tt.splat [[arg5_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.muli [[VAR_4_1_]], [[VAR_5_1_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_7_1_:%.+]] = tt.broadcast [[VAR_3_1_]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:             [[VAR_8_1_:%.+]] = tt.broadcast [[VAR_6_1_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.addi [[VAR_7_1_]], [[VAR_8_1_]] : tensor<2x2xi32>
// CHECK-DAG:         [[VAR_10_1_:%.+]] = tt.splat [[arg0_]] : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_1_:%.+]] = tt.addptr [[VAR_10_1_]], [[VAR_9_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_12_1_:%.+]] = tt.splat [[arg6_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_1_:%.+]] = arith.muli [[VAR_12_1_]], [[VAR_1_1_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_14_1_:%.+]] = tt.splat [[arg1_]] : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_15_1_:%.+]] = tt.addptr [[VAR_14_1_]], [[VAR_13_1_]] : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
// CHECK-DAG:         [[VAR_16_1_:%.+]] = tt.splat [[arg7_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_1_:%.+]] = arith.muli [[VAR_16_1_]], [[VAR_4_1_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_18_1_:%.+]] = tt.broadcast [[VAR_15_1_]] : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
// CHECK:             [[VAR_19_1_:%.+]] = tt.broadcast [[VAR_17_1_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = tt.addptr [[VAR_18_1_]], [[VAR_19_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_21_1_:%.+]] = arith.muli [[arg5_]], [[CST_32_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_1_:%.+]] = tt.splat [[VAR_21_1_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_23_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_11_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:         [[VAR_24_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_20_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_1_:%.+]]:10 = scf.for [[VAR_arg8_1_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]] step [[CST_1_1_]] iter_args([[VAR_arg9_1_:%.+]] = [[VAR_23_1_]]#0, [[VAR_arg10_1_:%.+]] = [[VAR_23_1_]]#1, [[VAR_arg11_1_:%.+]] = [[VAR_23_1_]]#2, [[VAR_arg12_1_:%.+]] = [[VAR_23_1_]]#3, [[VAR_arg13_1_:%.+]] = [[VAR_23_1_]]#4, [[VAR_arg14_1_:%.+]] = [[VAR_24_1_]]#0, [[VAR_arg15_1_:%.+]] = [[VAR_24_1_]]#1, [[VAR_arg16_1_:%.+]] = [[VAR_24_1_]]#2, [[VAR_arg17_1_:%.+]] = [[VAR_24_1_]]#3, [[VAR_arg18_1_:%.+]] = [[VAR_24_1_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:           [[LOAD_VAR_arg9_MEM_1_:%.+]] = tt.load [[VAR_arg9_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:               tt.store [[VAR_arg14_1_]], [[LOAD_VAR_arg9_MEM_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = tt.addptr [[VAR_arg9_1_]], [[VAR_22_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = tt.addptr [[VAR_arg14_1_]], [[VAR_22_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_27_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:           [[VAR_30_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_28_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]]:10 = scf.for [[VAR_arg19_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]] step [[CST_1_1_]] iter_args([[VAR_arg20_:%.+]] = [[VAR_29_1_]]#0, [[VAR_arg21_:%.+]] = [[VAR_29_1_]]#1, [[VAR_arg22_:%.+]] = [[VAR_29_1_]]#2, [[VAR_arg23_:%.+]] = [[VAR_29_1_]]#3, [[VAR_arg24_:%.+]] = [[VAR_29_1_]]#4, [[VAR_arg25_:%.+]] = [[VAR_30_1_]]#0, [[VAR_arg26_:%.+]] = [[VAR_30_1_]]#1, [[VAR_arg27_:%.+]] = [[VAR_30_1_]]#2, [[VAR_arg28_:%.+]] = [[VAR_30_1_]]#3, [[VAR_arg29_:%.+]] = [[VAR_30_1_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:             [[LOAD_VAR_arg20_MEM_:%.+]] = tt.load [[VAR_arg20_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:                 tt.store [[VAR_arg25_]], [[LOAD_VAR_arg20_MEM_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[VAR_33_:%.+]] = tt.addptr [[VAR_arg20_]], [[VAR_22_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:             [[VAR_34_:%.+]] = tt.addptr [[VAR_arg25_]], [[VAR_22_1_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_35_:%.+]]:5 = "tts.get_structured_state"([[VAR_33_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:             [[VAR_36_:%.+]]:5 = "tts.get_structured_state"([[VAR_34_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:                 scf.yield [[VAR_35_]]#0, [[VAR_35_]]#1, [[VAR_35_]]#2, [[VAR_35_]]#3, [[VAR_35_]]#4, [[VAR_36_]]#0, [[VAR_36_]]#1, [[VAR_36_]]#2, [[VAR_36_]]#3, [[VAR_36_]]#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:               }
// CHECK:               scf.yield [[VAR_31_]]#0, [[VAR_31_]]#1, [[VAR_31_]]#2, [[VAR_31_]]#3, [[VAR_31_]]#4, [[VAR_31_]]#5, [[VAR_31_]]#6, [[VAR_31_]]#7, [[VAR_31_]]#8, [[VAR_31_]]#9 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:             }
// CHECK:             tt.return
// CHECK:           }

// ----

module {
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
        %25 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %25 : tensor<2x2x!tt.ptr<f32>>
        %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %27 = tt.addptr %arg9, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %26, %27 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %23 = tt.addptr %22#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %22#1, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %24 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:           tt.func public @nested2_use_loop_results([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:         [[CST_2_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:         [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:         [[VAR_0_2_:%.+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_1_2_:%.+]] = tt.expand_dims [[VAR_0_2_]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG:         [[VAR_2_2_:%.+]] = tt.splat [[arg2_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_2_:%.+]] = arith.muli [[VAR_1_2_]], [[VAR_2_2_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_4_2_:%.+]] = tt.expand_dims [[VAR_0_2_]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK-DAG:         [[VAR_5_2_:%.+]] = tt.splat [[arg3_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_2_:%.+]] = arith.muli [[VAR_4_2_]], [[VAR_5_2_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_7_2_:%.+]] = tt.broadcast [[VAR_3_2_]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:             [[VAR_8_2_:%.+]] = tt.broadcast [[VAR_6_2_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_9_2_:%.+]] = arith.addi [[VAR_7_2_]], [[VAR_8_2_]] : tensor<2x2xi32>
// CHECK-DAG:         [[VAR_10_2_:%.+]] = tt.splat [[arg0_]] : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_2_:%.+]] = tt.addptr [[VAR_10_2_]], [[VAR_9_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_12_2_:%.+]] = tt.splat [[arg1_]] : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
// CHECK:             [[VAR_13_2_:%.+]] = tt.addptr [[VAR_12_2_]], [[VAR_3_2_]] : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
// CHECK:             [[VAR_14_2_:%.+]] = tt.broadcast [[VAR_13_2_]] : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_15_2_:%.+]] = tt.addptr [[VAR_14_2_]], [[VAR_8_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_16_2_:%.+]] = arith.muli [[arg3_]], [[CST_4_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_2_:%.+]] = tt.splat [[VAR_16_2_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_18_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_11_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:         [[VAR_19_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_15_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_2_:%.+]]:10 = scf.for [[VAR_arg4_:%.+]] = [[CST_0_2_]] to [[CST_2_2_]] step [[CST_1_2_]] iter_args([[VAR_arg5_:%.+]] = [[VAR_18_2_]]#0, [[VAR_arg6_:%.+]] = [[VAR_18_2_]]#1, [[VAR_arg7_:%.+]] = [[VAR_18_2_]]#2, [[VAR_arg8_2_:%.+]] = [[VAR_18_2_]]#3, [[VAR_arg9_2_:%.+]] = [[VAR_18_2_]]#4, [[VAR_arg10_2_:%.+]] = [[VAR_19_2_]]#0, [[VAR_arg11_2_:%.+]] = [[VAR_19_2_]]#1, [[VAR_arg12_2_:%.+]] = [[VAR_19_2_]]#2, [[VAR_arg13_2_:%.+]] = [[VAR_19_2_]]#3, [[VAR_arg14_2_:%.+]] = [[VAR_19_2_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:           [[VAR_21_1_:%.+]] = tt.load [[VAR_arg5_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:               tt.store [[VAR_arg10_2_]], [[VAR_21_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_22_2_:%.+]] = tt.addptr [[VAR_arg5_]], [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:           [[VAR_23_2_:%.+]] = tt.addptr [[VAR_arg10_2_]], [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_22_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:           [[VAR_25_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_23_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_arg9_MEM_1_:%.+]]:10 = scf.for [[VAR_arg15_2_:%.+]] = [[CST_0_2_]] to [[CST_2_2_]] step [[CST_1_2_]] iter_args([[VAR_arg16_2_:%.+]] = [[VAR_24_2_]]#0, [[VAR_arg17_2_:%.+]] = [[VAR_24_2_]]#1, [[VAR_arg18_2_:%.+]] = [[VAR_24_2_]]#2, [[VAR_arg19_1_:%.+]] = [[VAR_24_2_]]#3, [[VAR_arg20_1_:%.+]] = [[VAR_24_2_]]#4, [[VAR_arg21_1_:%.+]] = [[VAR_25_2_]]#0, [[VAR_arg22_1_:%.+]] = [[VAR_25_2_]]#1, [[VAR_arg23_1_:%.+]] = [[VAR_25_2_]]#2, [[VAR_arg24_1_:%.+]] = [[VAR_25_2_]]#3, [[VAR_arg25_1_:%.+]] = [[VAR_25_2_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:             [[VAR_31_1_:%.+]] = tt.load [[VAR_arg16_2_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:                 tt.store [[VAR_arg21_1_]], [[VAR_31_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[LOAD_VAR_arg20_MEM_1_:%.+]] = tt.addptr [[VAR_arg16_2_]], [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:             [[VAR_33_1_:%.+]] = tt.addptr [[VAR_arg21_1_]], [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_1_:%.+]]:5 = "tts.get_structured_state"([[LOAD_VAR_arg20_MEM_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:             [[VAR_35_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_33_1_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:                 scf.yield [[VAR_34_1_]]#0, [[VAR_34_1_]]#1, [[VAR_34_1_]]#2, [[VAR_34_1_]]#3, [[VAR_34_1_]]#4, [[VAR_35_1_]]#0, [[VAR_35_1_]]#1, [[VAR_35_1_]]#2, [[VAR_35_1_]]#3, [[VAR_35_1_]]#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:               }
// CHECK-DAG:           [[VAR_27_2_:%.+]] = tt.addptr [[LOAD_VAR_arg9_MEM_1_]]#0, [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:           [[VAR_28_2_:%.+]] = tt.addptr [[LOAD_VAR_arg9_MEM_1_]]#5, [[VAR_17_2_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_27_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:           [[VAR_30_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_28_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:               scf.yield [[VAR_29_2_]]#0, [[VAR_29_2_]]#1, [[VAR_29_2_]]#2, [[VAR_29_2_]]#3, [[VAR_29_2_]]#4, [[VAR_30_2_]]#0, [[VAR_30_2_]]#1, [[VAR_30_2_]]#2, [[VAR_30_2_]]#3, [[VAR_30_2_]]#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:             }
// CHECK:             tt.return
// CHECK:           }

// ----

module {
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

// CHECK:           tt.func public @nested3([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:         [[CST_2_3_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:         [[VAR_0_3_:%.+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_1_3_:%.+]] = tt.expand_dims [[VAR_0_3_]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG:         [[VAR_2_3_:%.+]] = tt.splat [[arg2_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_3_:%.+]] = arith.muli [[VAR_1_3_]], [[VAR_2_3_]] : tensor<2x1xi32>
// CHECK-DAG:         [[VAR_4_3_:%.+]] = tt.expand_dims [[VAR_0_3_]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK-DAG:         [[VAR_5_3_:%.+]] = tt.splat [[arg3_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_3_:%.+]] = arith.muli [[VAR_4_3_]], [[VAR_5_3_]] : tensor<1x2xi32>
// CHECK-DAG:         [[VAR_7_3_:%.+]] = tt.broadcast [[VAR_3_3_]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:             [[VAR_8_3_:%.+]] = tt.broadcast [[VAR_6_3_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = arith.addi [[VAR_7_3_]], [[VAR_8_3_]] : tensor<2x2xi32>
// CHECK-DAG:         [[VAR_10_3_:%.+]] = tt.splat [[arg0_]] : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_3_:%.+]] = tt.addptr [[VAR_10_3_]], [[VAR_9_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_12_3_:%.+]] = tt.splat [[arg1_]] : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
// CHECK:             [[VAR_13_3_:%.+]] = tt.addptr [[VAR_12_3_]], [[VAR_3_3_]] : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
// CHECK:             [[VAR_14_3_:%.+]] = tt.broadcast [[VAR_13_3_]] : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_15_3_:%.+]] = tt.addptr [[VAR_14_3_]], [[VAR_8_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:         [[VAR_16_3_:%.+]] = arith.muli [[arg3_]], [[CST_2_3_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_3_:%.+]] = tt.splat [[VAR_16_3_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_18_3_:%.+]] = arith.muli [[arg3_]], [[CST_2_3_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_3_:%.+]] = tt.splat [[VAR_18_3_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:         [[VAR_20_3_:%.+]]:5 = "tts.get_structured_state"([[VAR_11_3_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-DAG:         [[VAR_21_2_:%.+]]:5 = "tts.get_structured_state"([[VAR_15_3_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_3_:%.+]]:10 = scf.for [[VAR_arg4_1_:%.+]] = [[CST_0_3_]] to [[CST_2_3_]] step [[CST_1_3_]] iter_args([[VAR_arg5_1_:%.+]] = [[VAR_20_3_]]#0, [[VAR_arg6_1_:%.+]] = [[VAR_20_3_]]#1, [[VAR_arg7_1_:%.+]] = [[VAR_20_3_]]#2, [[VAR_arg8_3_:%.+]] = [[VAR_20_3_]]#3, [[VAR_arg9_3_:%.+]] = [[VAR_20_3_]]#4, [[VAR_arg10_3_:%.+]] = [[VAR_21_2_]]#0, [[VAR_arg11_3_:%.+]] = [[VAR_21_2_]]#1, [[VAR_arg12_3_:%.+]] = [[VAR_21_2_]]#2, [[VAR_arg13_3_:%.+]] = [[VAR_21_2_]]#3, [[VAR_arg14_3_:%.+]] = [[VAR_21_2_]]#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:           [[VAR_23_2_:%.+]] = tt.load [[VAR_arg5_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_24_3_:%.+]]:10 = scf.for [[VAR_arg15_3_:%.+]] = [[CST_0_3_]] to [[CST_2_3_]] step [[CST_1_3_]] iter_args([[VAR_arg16_3_:%.+]] = [[VAR_arg5_1_]], [[VAR_arg17_3_:%.+]] = [[VAR_arg6_1_]], [[VAR_arg18_3_:%.+]] = [[VAR_arg7_1_]], [[VAR_arg19_2_:%.+]] = [[VAR_arg8_3_]], [[VAR_arg20_2_:%.+]] = [[VAR_arg9_3_]], [[VAR_arg21_2_:%.+]] = [[VAR_arg10_3_]], [[VAR_arg22_2_:%.+]] = [[VAR_arg11_3_]], [[VAR_arg23_2_:%.+]] = [[VAR_arg12_3_]], [[VAR_arg24_2_:%.+]] = [[VAR_arg13_3_]], [[VAR_arg25_2_:%.+]] = [[VAR_arg14_3_]]) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:             [[VAR_27_3_:%.+]] = tt.addptr [[VAR_arg16_3_]], [[VAR_17_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_28_2_:%.+]] = tt.load [[VAR_27_3_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[VAR_29_3_:%.+]]:5 = "tts.get_structured_state"([[VAR_27_3_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_30_3_:%.+]]:10 = scf.for [[VAR_arg26_1_:%.+]] = [[CST_0_3_]] to [[CST_2_3_]] step [[CST_1_3_]] iter_args([[VAR_arg27_1_:%.+]] = [[VAR_29_3_]]#0, [[VAR_arg28_1_:%.+]] = [[VAR_29_3_]]#1, [[VAR_arg29_1_:%.+]] = [[VAR_29_3_]]#2, [[VAR_arg30_:%.+]] = [[VAR_29_3_]]#3, [[VAR_arg31_:%.+]] = [[VAR_29_3_]]#4, [[VAR_arg32_:%.+]] = [[VAR_arg21_2_]], [[VAR_arg33_:%.+]] = [[VAR_arg22_2_]], [[VAR_arg34_:%.+]] = [[VAR_arg23_2_]], [[VAR_arg35_:%.+]] = [[VAR_arg24_2_]], [[VAR_arg36_:%.+]] = [[VAR_arg25_2_]]) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
// CHECK-DAG:               [[VAR_31_2_:%.+]] = tt.addptr [[VAR_arg27_1_]], [[VAR_17_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK:                   [[LOAD_VAR_arg20_MEM_1_:%.+]] = tt.load [[VAR_31_2_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:                   tt.store [[VAR_arg32_]], [[VAR_23_2_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:                   [[VAR_33_2_:%.+]] = tt.addptr [[VAR_arg32_]], [[VAR_17_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK:                   tt.store [[VAR_33_2_]], [[VAR_28_2_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:                   [[VAR_34_2_:%.+]] = tt.addptr [[VAR_33_2_]], [[VAR_17_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK:                   tt.store [[VAR_34_2_]], [[LOAD_VAR_arg20_MEM_1_]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:               [[VAR_35_2_:%.+]] = tt.addptr [[VAR_34_2_]], [[VAR_17_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-DAG:               [[VAR_36_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_31_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:                   [[VAR_37_:%.+]]:5 = "tts.get_structured_state"([[VAR_35_2_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:                   scf.yield [[VAR_36_1_]]#0, [[VAR_36_1_]]#1, [[VAR_36_1_]]#2, [[VAR_36_1_]]#3, [[VAR_36_1_]]#4, [[VAR_37_]]#0, [[VAR_37_]]#1, [[VAR_37_]]#2, [[VAR_37_]]#3, [[VAR_37_]]#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:                 }
// CHECK:                 scf.yield [[VAR_30_3_]]#0, [[VAR_30_3_]]#1, [[VAR_30_3_]]#2, [[VAR_30_3_]]#3, [[VAR_30_3_]]#4, [[VAR_30_3_]]#5, [[VAR_30_3_]]#6, [[VAR_30_3_]]#7, [[VAR_30_3_]]#8, [[VAR_30_3_]]#9 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:               }
// CHECK:               [[VAR_25_3_:%.+]] = tt.addptr [[VAR_24_3_]]#0, [[VAR_19_3_]] : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK:               [[LOAD_VAR_arg9_MEM_1_1_:%.+]]:5 = "tts.get_structured_state"([[VAR_25_3_]]) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
// CHECK:               scf.yield [[LOAD_VAR_arg9_MEM_1_1_]]#0, [[LOAD_VAR_arg9_MEM_1_1_]]#1, [[LOAD_VAR_arg9_MEM_1_1_]]#2, [[LOAD_VAR_arg9_MEM_1_1_]]#3, [[LOAD_VAR_arg9_MEM_1_1_]]#4, [[VAR_24_3_]]#5, [[VAR_24_3_]]#6, [[VAR_24_3_]]#7, [[VAR_24_3_]]#8, [[VAR_24_3_]]#9 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
// CHECK:             }
// CHECK:             tt.return
// CHECK:           }
