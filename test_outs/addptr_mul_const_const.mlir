module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32) {
    %c11 = arith.constant 11 : index
    %c20480 = arith.constant 20480 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.addi %2, %c20480 : index
    %4 = tts.make_tptr %arg0 to sizes: [1024], strides: [%c11], offsets: [%3], shape: [0], order: [] : <bf16, 1> to tensor<1024x!tt.ptr<bf16, 1>>
    %5 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <bf16, 1> to tensor<1024x!tt.ptr<bf16, 1>>
    %6 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16, 1>>) -> tensor<1024xbf16>
    "tts.store"(%5, %6) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16, 1>>, tensor<1024xbf16>) -> ()
    tt.return
  }
}

