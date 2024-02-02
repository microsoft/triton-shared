module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: i32) {
    %c256 = arith.constant 256 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<4x256xbf16>
    %c1 = arith.constant 1 : index
    %c256_i32 = arith.constant 256 : i32
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<1x256x!tt.ptr<bf16, 1>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16, 1>>) -> tensor<1x256xbf16>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [%2, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<1x256x!tt.ptr<bf16, 1>>
    "tts.store"(%3, %1) <{static_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16, 1>>, tensor<1x256xbf16>) -> ()
    %4:2 = scf.for %arg2 = %c0 to %c12 step %c3 iter_args(%arg3 = %cst, %arg4 = %c0) -> (tensor<4x256xbf16>, index) {
      %6 = arith.index_cast %arg2 : index to i32
      %7 = arith.muli %6, %c256_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%8, %c1], offsets: [%arg4, %c0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
      %10 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
      %11 = arith.addf %arg3, %10 : tensor<4x256xbf16>
      %12 = arith.addi %arg4, %c256 : index
      scf.yield %11, %12 : tensor<4x256xbf16>, index
    }
    %5 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    "tts.store"(%5, %4#0) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

