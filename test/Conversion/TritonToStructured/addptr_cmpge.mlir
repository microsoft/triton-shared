// RUN: triton-shared-opt --triton-to-structured --split-input-file %s | FileCheck %s

// These tests check that loads/stores that exhibit a cmp ge against 0 work correctly with the pointer analysis pass

tt.func public @test_masked_load(%arg0: !tt.ptr<f16>) -> tensor<16x16xf16> {
  %cst = arith.constant dense<0> : tensor<1x16xi64>
  %c16_i32 = arith.constant 16 : i32
  %0 = tt.get_program_id y : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = arith.extsi %1 : i32 to i64
  %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %5 = arith.extsi %4 : tensor<16xi32> to tensor<16xi64>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi64> -> tensor<16x16xi64>
  %8 = tt.addptr %3, %7 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %9 = tt.splat %2 : i64 -> tensor<16xi64>
  %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %11 = arith.extsi %10 : tensor<16xi32> to tensor<16xi64>
  %12 = arith.addi %9, %11 : tensor<16xi64>
  %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %14 = arith.cmpi sge, %13, %cst : tensor<1x16xi64>
  %15 = tt.broadcast %14 : tensor<1x16xi1> -> tensor<16x16xi1>
  %16 = tt.load %8, %15 evictionPolicy = evict_last : tensor<16x16x!tt.ptr<f16>>
  tt.return %16 : tensor<16x16xf16>
}

// CHECK:         tt.func public @test_masked_load([[arg0_:%.+]]: !tt.ptr<f16>) -> tensor<16x16xf16> {
// CHECK:           [[VAR_0_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [16, 16], strides: [1, 0], offsets: [0, 0], shape: [0, 0], order: [] : <f16> to tensor<16x16x!tt.ptr<f16>>
// CHECK:           [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64: 16, 16>}> : (tensor<16x16x!tt.ptr<f16>>) -> tensor<16x16xf16>
// CHECK:         }

// -----

tt.func public @test_masked_store(%arg0: !tt.ptr<f16>) {
  %cst = arith.constant dense<0> : tensor<16x1xi64>
  %cst_0 = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %2 = arith.extsi %1 : tensor<16xi32> to tensor<16xi64>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %4 = tt.broadcast %3 : tensor<16x1xi64> -> tensor<16x16xi64>
  %5 = tt.addptr %0, %4 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %6 = arith.cmpi sge, %3, %cst : tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi1> -> tensor<16x16xi1>
  tt.store %5, %cst_0, %7 : tensor<16x16x!tt.ptr<f16>>
  tt.return
}

// CHECK:         tt.func public @test_masked_store([[arg0_:%.+]]: !tt.ptr<f16>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [16, 16], strides: [1, 0], offsets: [0, 0], shape: [0, 0], order: [] : <f16> to tensor<16x16x!tt.ptr<f16>>
// CHECK:           "tts.store"([[VAR_0_]], [[VAR_cst_]]) <{static_mask_dims = array<i64: 16, 16>}> : (tensor<16x16x!tt.ptr<f16>>, tensor<16x16xf16>) -> ()
// CHECK:         }
