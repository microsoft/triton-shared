module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32) {
    %cst = arith.constant 0xFF80 : bf16
    %c128 = arith.constant 128 : index
    %0 = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <bf16, 1> to tensor<128x!tt.ptr<bf16, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <bf16, 1> to tensor<128x!tt.ptr<bf16, 1>>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.minsi %2, %c128 : index
    %4 = "tts.load"(%0, %3, %cst) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<bf16, 1>>, index, bf16) -> tensor<128xbf16>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.minsi %5, %c128 : index
    "tts.store"(%1, %4, %6) <{static_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<bf16, 1>>, tensor<128xbf16>, index) -> ()
    tt.return
  }
}

