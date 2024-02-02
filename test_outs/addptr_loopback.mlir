module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32) {
    %c6 = arith.constant 6 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c6], offsets: [%0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = tts.make_tptr %arg1 to sizes: [4, 256], strides: [1, %c6], offsets: [%2, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %4 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    "tts.store"(%3, %4) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

