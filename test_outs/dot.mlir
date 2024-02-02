module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>) {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %0 = tts.make_tptr %arg0 to sizes: [128, 64], strides: [%c128, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x64x!tt.ptr<bf16, 1>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16, 1>>) -> tensor<128x64xbf16>
    %2 = tts.make_tptr %arg1 to sizes: [256, 64], strides: [1, %c256], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<256x64x!tt.ptr<bf16, 1>>
    %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x64x!tt.ptr<bf16, 1>>) -> tensor<256x64xbf16>
    %4 = tt.trans %3 : (tensor<256x64xbf16>) -> tensor<64x256xbf16>
    %5 = tts.make_tptr %arg2 to sizes: [128, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %6 = "tts.load"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>) -> tensor<128x256xbf16>
    %7 = tt.dot %1, %4, %6 {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    "tts.store"(%5, %7) <{static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>) -> ()
    tt.return
  }
}

