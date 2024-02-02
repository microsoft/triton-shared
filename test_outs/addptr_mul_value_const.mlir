module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32) {
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %arg2 : i32 to index
    %4 = arith.muli %3, %c2048 : index
    %5 = arith.addi %2, %4 : index
    %6 = arith.addi %3, %c1 : index
    %7 = tts.make_tptr %arg0 to sizes: [1024], strides: [%6], offsets: [%5], shape: [0], order: [] : <bf16, 1> to tensor<1024x!tt.ptr<bf16, 1>>
    %8 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <bf16, 1> to tensor<1024x!tt.ptr<bf16, 1>>
    %9 = "tts.load"(%7) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16, 1>>) -> tensor<1024xbf16>
    "tts.store"(%8, %9) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<bf16, 1>>, tensor<1024xbf16>) -> ()
    tt.return
  }
}

