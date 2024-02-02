module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = tts.make_tptr %arg1 to sizes: [1024, 1024], strides: [1, 1], offsets: [%2, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<1024x1024x!tt.ptr<f32, 1>>
    %4 = "tts.load"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x1024x!tt.ptr<f32, 1>>) -> tensor<1024x1024xf32>
    %5 = math.exp %4 : tensor<1024x1024xf32>
    %6 = arith.muli %0, %arg3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = tts.make_tptr %arg0 to sizes: [1024, 1024], strides: [1, 1], offsets: [%7, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<1024x1024x!tt.ptr<f32, 1>>
    "tts.store"(%8, %5) <{static_dims = array<i64>}> : (tensor<1024x1024x!tt.ptr<f32, 1>>, tensor<1024x1024xf32>) -> ()
    tt.return
  }
}

