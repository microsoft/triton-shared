module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %0 = tts.make_tptr %arg0 to sizes: [128, 64], strides: [%c128, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x64x!tt.ptr<bf16, 1>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16, 1>>) -> tensor<128x64xbf16>
    %2 = tts.make_tptr %arg1 to sizes: [64, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<64x256x!tt.ptr<bf16, 1>>
    %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<64x256x!tt.ptr<bf16, 1>>) -> tensor<64x256xbf16>
    %4 = tts.make_tptr %arg2 to sizes: [128, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %5 = tt.dot %1, %3, %cst {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    "tts.store"(%4, %5) <{static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>) -> ()
    "tts.store"(%4, %cst) <{static_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>) -> ()
    tt.return
  }
}

