module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: tensor<1024x!tt.ptr<f32, 1>>) {
    %0 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %2 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
    %3 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
    %4 = arith.addf %2, %3 : tensor<1024xf32>
    %5 = arith.subf %4, %3 : tensor<1024xf32>
    %6 = arith.mulf %5, %3 : tensor<1024xf32>
    %7 = arith.divf %6, %3 : tensor<1024xf32>
    %8 = arith.cmpf oeq, %7, %3 : tensor<1024xf32>
    %9 = arith.select %8, %2, %3 : tensor<1024xi1>, tensor<1024xf32>
    tt.store %arg2, %9 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }
}

