module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<i32, 1>) {
    %c6144 = arith.constant 6144 : index
    %c6 = arith.constant 6 : index
    %0 = tt.make_range {end = 768 : i32, start = 512 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
    %2 = tt.broadcast %1 : (tensor<256x1xi32>) -> tensor<256x128xi32>
    %3 = tts.make_tptr %arg1 to sizes: [256, 128], strides: [1, %c6], offsets: [512, %c6144], shape: [0, 0], order: [] : <bf16, 1> to tensor<256x128x!tt.ptr<bf16, 1>>
    %4 = "tts.load"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>) -> tensor<256x128xbf16>
    "tts.store"(%3, %4) <{static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>, tensor<256x128xbf16>) -> ()
    %5 = tts.make_tptr %arg2 to sizes: [256, 128], strides: [1, %c6], offsets: [512, %c6144], shape: [0, 0], order: [] : <i32, 1> to tensor<256x128x!tt.ptr<i32, 1>>
    "tts.store"(%5, %2) <{static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<i32, 1>>, tensor<256x128xi32>) -> ()
    tt.return
  }
}

