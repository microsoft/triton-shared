module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>) {
    %c6144 = arith.constant 6144 : index
    %c6 = arith.constant 6 : index
    %0 = tts.make_tptr %arg1 to sizes: [256, 128], strides: [1, %c6], offsets: [512, %c6144], shape: [0, 0], order: [] : <bf16, 1> to tensor<256x128x!tt.ptr<bf16, 1>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>) -> tensor<256x128xbf16>
    "tts.store"(%0, %1) <{static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>, tensor<256x128xbf16>) -> ()
    tt.return
  }
}

