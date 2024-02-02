module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg12 : i32 to index
    %2 = arith.muli %1, %0 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg7 : i32 to index
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = arith.muli %4, %c64 : index
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %cst, %arg16 = %6, %arg17 = %2) -> (tensor<128x64xbf16>, index, index)  : i32 {
      %18 = tts.make_tptr %arg0 to sizes: [128, 64], strides: [%0, %4], offsets: [%arg17, %c0], shape: [%3, %5], order: [1, 0] : <bf16, 1> to !tt.ptr<tensor<128x64xbf16>, 1>
      %19 = tts.make_tptr %arg0 to sizes: [128, 64], strides: [%0, %4], offsets: [%2, %arg16], shape: [%3, %5], order: [1, 0] : <bf16, 1> to !tt.ptr<tensor<128x64xbf16>, 1>
      %20 = "tts.load"(%19) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>, 1>) -> tensor<128x64xbf16>
      %21 = "tts.load"(%18) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>, 1>) -> tensor<128x64xbf16>
      %22 = arith.addf %20, %21 : tensor<128x64xbf16>
      %23 = arith.addf %arg15, %22 : tensor<128x64xbf16>
      %24 = arith.muli %4, %c64 : index
      %25 = arith.addi %24, %arg16 : index
      %26 = arith.muli %0, %c64 : index
      %27 = arith.addi %26, %arg17 : index
      scf.yield %23, %25, %27 : tensor<128x64xbf16>, index, index
    }
    %8 = arith.muli %arg13, %c256_i32 : i32
    %9 = arith.index_cast %arg10 : i32 to index
    %10 = arith.index_cast %arg12 : i32 to index
    %11 = arith.muli %10, %9 : index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.index_cast %arg11 : i32 to index
    %14 = arith.index_cast %8 : i32 to index
    %15 = arith.muli %14, %13 : index
    %16 = arith.index_cast %arg4 : i32 to index
    %17 = tts.make_tptr %arg2 to sizes: [128, 64], strides: [%9, %13], offsets: [%11, %15], shape: [%12, %16], order: [1, 0] : <bf16, 1> to !tt.ptr<tensor<128x64xbf16>, 1>
    "tts.store"(%17, %7#0) <{static_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>, 1>, tensor<128x64xbf16>) -> ()
    tt.return
  }
}

