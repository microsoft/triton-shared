module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: tensor<128x128x!tt.ptr<f32, 1>>, %arg3: tensor<128x128x!tt.ptr<f32, 1>>) {
    %0 = tts.make_tptr %arg0 to sizes: [128, 128], strides: [1, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [128, 128], strides: [1, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    %2 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
    %3 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
    %4 = arith.addf %2, %3 : tensor<128x128xf32>
    %5 = arith.subf %2, %3 : tensor<128x128xf32>
    tt.store %arg2, %4 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.store %arg3, %5 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.return
  }
}

