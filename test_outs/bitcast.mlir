module {
  tt.func @kernel(%arg0: !tt.ptr<i32, 1>, %arg1: !tt.ptr<f32, 1>) {
    %0 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<1024x!tt.ptr<i32, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %2 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32, 1>>) -> tensor<1024xi32>
    %3 = tt.bitcast %2 : tensor<1024xi32> -> tensor<1024xf32>
    "tts.store"(%1, %3) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>) -> ()
    tt.return
  }
}

