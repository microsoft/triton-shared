module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.muli %0, %arg3 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.addi %2, %4 : index
    %6 = arith.muli %0, %arg4 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.addi %5, %7 : index
    %9 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%8], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %10 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
    %11 = math.exp %10 : tensor<1024xf32>
    %12 = arith.muli %0, %arg3 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%13], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    "tts.store"(%14, %11) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>) -> ()
    tt.return
  }
}

