module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %c128 = arith.constant 128 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.addi %2, %c128 : index
    %4 = tts.make_tptr %arg1 to sizes: [128, 128], strides: [1, 1], offsets: [%3, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
    %6 = math.exp %5 : tensor<128x128xf32>
    %7 = arith.muli %0, %arg3 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.addi %8, %c128 : index
    %10 = tts.make_tptr %arg0 to sizes: [128, 128], strides: [1, 1], offsets: [%9, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    "tts.store"(%10, %6) <{static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>, tensor<128x128xf32>) -> ()
    tt.return
  }
}

