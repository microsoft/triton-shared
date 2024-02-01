// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : (i32) -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : (tensor<128x1xi32>) -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : (tensor<1x64xi32>) -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
    %12 = tt.broadcast %11 : (tensor<256x1xi32>) -> tensor<256x64xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : (i32) -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %17 = tt.broadcast %16 : (tensor<1x64xi32>) -> tensor<256x64xi32>
    %18 = arith.addi %12, %17 : tensor<256x64xi32>
    %20 = tt.splat %c256 : (i32) -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : (tensor<128x1xi32>) -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : (tensor<1x256xi32>) -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64xbf16>
    %40 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<256x64x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x64xbf16>
    %43 = tt.trans %42 : (tensor<256x64xbf16>) -> tensor<64x256xbf16>
    %50 = tt.splat %arg2 : (!tt.ptr<bf16>) -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %52 = tt.load %51 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x256xbf16>
    %60 = tt.dot %32, %43, %52 {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256xbf16>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_1_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_2_:%.+]]: !tt.ptr<bf16, 1>) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK:           [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 64], strides: {{.}}[[CST_128_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x64x!tt.ptr<bf16, 1>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16, 1>>) -> tensor<128x64xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [256, 64], strides: [1, [[CST_256_]]{{.}}, offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<256x64x!tt.ptr<bf16, 1>>
// CHECK:           [[VAR_3_:%.+]] = "tts.load"([[VAR_2_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x64x!tt.ptr<bf16, 1>>) -> tensor<256x64xbf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tt.trans [[VAR_3_]] : (tensor<256x64xbf16>) -> tensor<64x256xbf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [128, 256], strides: {{.}}[[CST_256_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
// CHECK:           [[VAR_6_:%.+]] = "tts.load"([[VAR_5_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>) -> tensor<128x256xbf16>
// CHECK:           [[VAR_7_:%.+]] = tt.dot [[VAR_1_]], [[VAR_4_]], [[VAR_6_]] {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
// CHECK:           "tts.store"([[VAR_5_]], [[VAR_7_]]) <{static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
