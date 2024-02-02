module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: i32) {
    %c15 = arith.constant 15 : index
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c5], offsets: [%0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.addi %0, %3 : index
    %5 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [2, %c10], offsets: [%4, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %6 = "tts.load"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    %7 = arith.addf %2, %6 : tensor<4x256xbf16>
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.addi %4, %8 : index
    %10 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [3, %c15], offsets: [%9, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    "tts.store"(%10, %7) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

