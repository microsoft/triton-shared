module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>, %arg3: i32) {
    %c5 = arith.constant 5 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c5], offsets: [%0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = tts.make_tptr %arg1 to sizes: [4, 256], strides: [1, %c5], offsets: [%3, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    %6 = arith.addf %2, %5 : tensor<4x256xbf16>
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = tts.make_tptr %arg2 to sizes: [4, 256], strides: [1, %c5], offsets: [%7, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    "tts.store"(%8, %6) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

