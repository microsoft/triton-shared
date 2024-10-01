// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @scf_if(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<bf16>, %arg5: !tt.ptr<bf16>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<256x1xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xbf16>
    %cst_1 = arith.constant dense<128> : tensor<2048x1xi32>
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_2 = arith.constant dense<2048> : tensor<256x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : i32 -> tensor<128xi32>
    %5 = arith.addi %4, %3 : tensor<128xi32>
    %6 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %7 = tt.expand_dims %1 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %8 = arith.muli %7, %cst_2 : tensor<256x1xi32>
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<2048xi32> -> tensor<1x2048xi32>
    %10 = tt.broadcast %8 : tensor<256x1xi32> -> tensor<256x2048xi32>
    %11 = tt.broadcast %9 : tensor<1x2048xi32> -> tensor<256x2048xi32>
    %12 = arith.addi %10, %11 : tensor<256x2048xi32>
    %13 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x2048x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %12 : tensor<256x2048x!tt.ptr<bf16>>, tensor<256x2048xi32>
    %15 = arith.divsi %0, %c4_i32 : i32
    %16 = arith.cmpi eq, %15, %c0_i32 : i32
    %17 = scf.if %16 -> (!tt.ptr<bf16>) {
      scf.yield %arg1 : !tt.ptr<bf16>
    } else {
      %35 = arith.cmpi eq, %15, %c1_i32 : i32
      %36 = scf.if %35 -> (!tt.ptr<bf16>) {
        scf.yield %arg2 : !tt.ptr<bf16>
      } else {
        %37 = arith.cmpi eq, %15, %c2_i32 : i32
        %38 = arith.select %37, %arg3, %arg4 : !tt.ptr<bf16>
        scf.yield %38 : !tt.ptr<bf16>
      }
      scf.yield %36 : !tt.ptr<bf16>
    }
    %18 = tt.expand_dims %6 {axis = 1 : i32} : tensor<2048xi32> -> tensor<2048x1xi32>
    %19 = arith.muli %18, %cst_1 : tensor<2048x1xi32>
    %20 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %21 = tt.broadcast %19 : tensor<2048x1xi32> -> tensor<2048x128xi32>
    %22 = tt.broadcast %20 : tensor<1x128xi32> -> tensor<2048x128xi32>
    %23 = arith.addi %21, %22 : tensor<2048x128xi32>
    %24 = tt.splat %17 : !tt.ptr<bf16> -> tensor<2048x128x!tt.ptr<bf16>>
    %25 = tt.addptr %24, %23 : tensor<2048x128x!tt.ptr<bf16>>, tensor<2048x128xi32>
    %26 = tt.load %25 : tensor<2048x128x!tt.ptr<bf16>>
    %27 = tt.load %14 : tensor<256x2048x!tt.ptr<bf16>>
    %28 = tt.dot %27, %26, %cst_0 : tensor<256x2048xbf16> * tensor<2048x128xbf16> -> tensor<256x128xbf16>
    %29 = arith.muli %7, %cst : tensor<256x1xi32>
    %30 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<256x1x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %29 : tensor<256x1x!tt.ptr<bf16>>, tensor<256x1xi32>
    %32 = tt.broadcast %31 : tensor<256x1x!tt.ptr<bf16>> -> tensor<256x128x!tt.ptr<bf16>>
    %33 = tt.broadcast %20 : tensor<1x128xi32> -> tensor<256x128xi32>
    %34 = tt.addptr %32, %33 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
    tt.store %34, %28 : tensor<256x128x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   tt.func public @scf_if(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<bf16>, %arg5: !tt.ptr<bf16>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {noinline = false} {
// CHECK:     %c128 = arith.constant 128 : index
// CHECK:     %c2048 = arith.constant 2048 : index
// CHECK:     %cst = arith.constant dense<0.000000e+00> : tensor<256x128xbf16>
// CHECK:     %c2_i32 = arith.constant 2 : i32
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %c4_i32 = arith.constant 4 : i32
// CHECK:     %c128_i32 = arith.constant 128 : i32
// CHECK:     %0 = tt.get_program_id x : i32
// CHECK:     %1 = arith.muli %0, %c128_i32 : i32
// CHECK:     %2 = arith.index_cast %1 : i32 to index
// CHECK:     %3 = arith.index_cast %1 : i32 to index
// CHECK:     %4 = tts.make_tptr %arg0 to sizes: [256, 2048], strides: [%c2048, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<256x2048x!tt.ptr<bf16>>
// CHECK:     %5 = arith.divsi %0, %c4_i32 : i32
// CHECK:     %6 = arith.cmpi eq, %5, %c0_i32 : i32
// CHECK:     %7 = scf.if %6 -> (!tt.ptr<bf16>) {
// CHECK:       scf.yield %arg1 : !tt.ptr<bf16>
// CHECK:     } else {
// CHECK:       %13 = arith.cmpi eq, %5, %c1_i32 : i32
// CHECK:       %14 = scf.if %13 -> (!tt.ptr<bf16>) {
// CHECK:         scf.yield %arg2 : !tt.ptr<bf16>
// CHECK:       } else {
// CHECK:         %15 = arith.cmpi eq, %5, %c2_i32 : i32
// CHECK:         %16 = arith.select %15, %arg3, %arg4 : !tt.ptr<bf16>
// CHECK:         scf.yield %16 : !tt.ptr<bf16>
// CHECK:       }
// CHECK:       scf.yield %14 : !tt.ptr<bf16>
// CHECK:     }
// CHECK:     %8 = tts.make_tptr %7 to sizes: [2048, 128], strides: [%c128, 1], offsets: [0, %3], shape: [0, 0], order: [] : <bf16> to tensor<2048x128x!tt.ptr<bf16>>
// CHECK:     %9 = "tts.load"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2048x128x!tt.ptr<bf16>>) -> tensor<2048x128xbf16>
// CHECK:     %10 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<256x2048x!tt.ptr<bf16>>) -> tensor<256x2048xbf16>
// CHECK:     %11 = tt.dot %10, %9, %cst : tensor<256x2048xbf16> * tensor<2048x128xbf16> -> tensor<256x128xbf16>
// CHECK:     %12 = tts.make_tptr %arg5 to sizes: [256, 128], strides: [%c128, 1], offsets: [0, %2], shape: [0, 0], order: [] : <bf16> to tensor<256x128x!tt.ptr<bf16>>
// CHECK:     "tts.store"(%12, %11) <{static_mask_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xbf16>) -> ()
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
