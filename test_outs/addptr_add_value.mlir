module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32, %arg3: i32) {
    %c6 = arith.constant 6 : index
    %c10 = arith.constant 10 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.addi %0, %1 : index
    %3 = arith.addi %2, %c10 : index
    %4 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c6], offsets: [%3, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.addi %5, %6 : index
    %8 = arith.addi %7, %c10 : index
    %9 = tts.make_tptr %arg1 to sizes: [4, 256], strides: [1, %c6], offsets: [%8, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %10 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    "tts.store"(%9, %10) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

