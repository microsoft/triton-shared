// RUN: triton-shared-opt --triton-to-structured %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i8>>
    %4 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<i32>>
    %7 = arith.extsi %3 : tensor<128xi8> to tensor<128xi32>
    %8 = arith.cmpi eq, %7, %6 : tensor<128xi32>
    %9 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %10 = tt.addptr %9, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %11 = tt.bitcast %10 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %12 = arith.extui %8 : tensor<128xi1> to tensor<128xi8>
    tt.store %11, %12 : tensor<128x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   tt.func public @kernel(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:     [[VAR_0:%.+]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK:     [[VAR_1:%.+]] = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
// CHECK:     [[VAR_2:%.+]] = tts.make_tptr %arg1 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <i8> to tensor<128x!tt.ptr<i8>>
// CHECK:     [[VAR_3:%.+]] = tt.addptr [[VAR_1]], [[VAR_0]] : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
// CHECK:     [[VAR_4:%.+]] = "tts.load"([[VAR_2]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x!tt.ptr<i8>>) -> tensor<128xi8>
// CHECK:     [[VAR_5:%.+]] = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[VAR_6:%.+]] = tts.make_tptr %arg2 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<128x!tt.ptr<i32>>
// CHECK:     [[VAR_7:%.+]] = tt.addptr [[VAR_5]], [[VAR_0]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:     [[VAR_8:%.+]] = "tts.load"([[VAR_6]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x!tt.ptr<i32>>) -> tensor<128xi32>
// CHECK:     [[VAR_9:%.+]] = arith.extsi [[VAR_4]] : tensor<128xi8> to tensor<128xi32>
// CHECK:     [[VAR_10:%.+]] = arith.cmpi eq, [[VAR_9]], [[VAR_8]] : tensor<128xi32>
// CHECK:     [[VAR_11:%.+]] = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
// CHECK:     [[VAR_12:%.+]] = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <i1> to tensor<128x!tt.ptr<i1>>
// CHECK:     [[VAR_13:%.+]] = tt.addptr [[VAR_11]], [[VAR_0]] : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
// CHECK:     [[VAR_14:%.+]] = tts.cast_tptr [[VAR_12]] : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
// CHECK:     [[VAR_15:%.+]] = arith.extui [[VAR_10]] : tensor<128xi1> to tensor<128xi8>
// CHECK:     "tts.store"([[VAR_14]], [[VAR_15]]) <{static_mask_dims = array<i64>}> : (tensor<128x!tt.ptr<i8>>, tensor<128xi8>) -> ()
// CHECK:     tt.return
// CHECK:   }
// CHECK: }