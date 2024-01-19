// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    %6 = tt.splat %cst : (bf16) -> tensor<128x64xbf16>
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %6, %arg16 = %5, %arg17 = %4) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
      %13 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>> -> tensor<128x64xbf16>
      %14 = tt.load %arg17 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>> -> tensor<128x64xbf16>
      %15 = arith.addf %13, %14 : tensor<128x64xbf16>
      %16 = arith.addf %arg15, %15 : tensor<128x64xbf16>
      %17 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      %18 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      scf.yield %16, %17, %18 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
    }
    %8 = arith.extsi %arg10 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %arg13, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg2, [%0, %10], [%8, %9], [%arg12, %11] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %12, %7#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>, tensor<128x64xbf16>
    tt.return
  }
}

// CHECK:         tt.func public @matmul_kernel_with_block_pointers_01234567891011([[PARAM_0_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_1_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_2_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.extsi [[PARAM_3_]] : i32 to i64
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.extsi [[PARAM_5_]] : i32 to i64
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.extsi [[PARAM_6_]] : i32 to i64
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.extsi [[PARAM_7_]] : i32 to i64
// CHECK:           [[VAR_4_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, {{.}}[[PARAM_12_]], [[CST_0_]]{{.}} {order = array<i32: 1, 0>} : <tensor<128x64xbf16>, 1>
// CHECK:           [[VAR_5_:%.+]] = tt.advance [[VAR_4_]], {{.}}[[CST_0_]], [[CST_64_]]{{.}} : <tensor<128x64xbf16>, 1>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_]] to [[PARAM_5_]] step [[CST_64_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_cst_]], [[VAR_arg16_:%.+]] = [[VAR_5_]], [[VAR_arg17_:%.+]] = [[VAR_4_]]) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>, 1>, !tt.ptr<tensor<128x64xbf16>, 1>)  : i32 {
// CHECK-DAG:         [[LOAD_VAR_arg16_MEM_:%.+]] = tt.load [[VAR_arg16_]] {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>, 1> -> tensor<128x64xbf16>
// CHECK-DAG:         [[LOAD_VAR_arg17_MEM_:%.+]] = tt.load [[VAR_arg17_]] {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>, 1> -> tensor<128x64xbf16>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_VAR_arg16_MEM_]], [[LOAD_VAR_arg17_MEM_]] : tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addf [[VAR_arg15_]], [[VAR_14_]] : tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_16_:%.+]] = tt.advance [[VAR_arg16_]], {{.}}[[CST_0_]], [[CST_64_]]{{.}} : <tensor<128x64xbf16>, 1>
// CHECK-DAG:         [[VAR_17_:%.+]] = tt.advance [[VAR_arg17_]], {{.}}[[CST_64_]], [[CST_0_]]{{.}} : <tensor<128x64xbf16>, 1>
// CHECK:             scf.yield [[VAR_15_]], [[VAR_16_]], [[VAR_17_]] : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>, 1>, !tt.ptr<tensor<128x64xbf16>, 1>
// CHECK:           }
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.extsi [[PARAM_10_]] : i32 to i64
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.extsi [[PARAM_11_]] : i32 to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.extsi [[PARAM_4_]] : i32 to i64
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.muli [[PARAM_13_]], [[CST_256_]] : i32
// CHECK:           [[VAR_11_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{.}}[[VAR_0_]], [[VAR_9_]]{{.}}, {{.}}[[VAR_7_]], [[VAR_8_]]{{.}}, {{.}}[[PARAM_12_]], [[VAR_10_]]{{.}} {order = array<i32: 1, 0>} : <tensor<128x64xbf16>, 1>
// CHECK:           tt.store [[VAR_11_]], [[VAR_6_]]#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>, 1>, tensor<128x64xbf16>
// CHECK:           tt.return
// CHECK:         }
