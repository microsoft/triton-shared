module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    %6:2 = "tts.state_placeholder"(%5) : (!tt.ptr<tensor<128x64xbf16>>) -> (!tt.ptr<tensor<128x64xbf16>>, index)
    %7:2 = "tts.state_placeholder"(%4) : (!tt.ptr<tensor<128x64xbf16>>) -> (!tt.ptr<tensor<128x64xbf16>>, index)
    %8:5 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %cst, %arg16 = %6#0, %arg17 = %6#1, %arg18 = %7#0, %arg19 = %7#1) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, index, !tt.ptr<tensor<128x64xbf16>>, index)  : i32 {
      %14 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
      %15 = tt.load %arg18 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
      %16 = arith.addf %14, %15 : tensor<128x64xbf16>
      %17 = arith.addf %arg15, %16 : tensor<128x64xbf16>
      %18 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      %19 = tt.advance %arg18, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      %20:2 = "tts.state_placeholder"(%18) : (!tt.ptr<tensor<128x64xbf16>>) -> (!tt.ptr<tensor<128x64xbf16>>, index)
      %21:2 = "tts.state_placeholder"(%19) : (!tt.ptr<tensor<128x64xbf16>>) -> (!tt.ptr<tensor<128x64xbf16>>, index)
      scf.yield %17, %20#0, %20#1, %21#0, %21#1 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, index, !tt.ptr<tensor<128x64xbf16>>, index
    }
    %9 = arith.extsi %arg10 : i32 to i64
    %10 = arith.extsi %arg11 : i32 to i64
    %11 = arith.extsi %arg4 : i32 to i64
    %12 = arith.muli %arg13, %c256_i32 : i32
    %13 = tt.make_tensor_ptr %arg2, [%0, %11], [%9, %10], [%arg12, %12] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %13, %8#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
    tt.return
  }
}