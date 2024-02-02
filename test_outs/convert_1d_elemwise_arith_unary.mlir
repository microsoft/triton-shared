module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<i32, 1>, %arg2: !tt.ptr<f16, 1>, %arg3: tensor<1024x!tt.ptr<bf16, 1>>, %arg4: tensor<1024x!tt.ptr<f32, 1>>, %arg5: tensor<1024x!tt.ptr<f32, 1>>, %arg6: tensor<1024x!tt.ptr<f32, 1>>, %arg7: tensor<1024x!tt.ptr<f32, 1>>) {
    %0 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<1024x!tt.ptr<i32, 1>>
    %2 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f16, 1> to tensor<1024x!tt.ptr<f16, 1>>
    %3 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
    %4 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32, 1>>) -> tensor<1024xi32>
    %5 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f16, 1>>) -> tensor<1024xf16>
    %6 = arith.truncf %3 : tensor<1024xf32> to tensor<1024xbf16>
    %7 = math.exp %3 : tensor<1024xf32>
    %8 = arith.sitofp %4 : tensor<1024xi32> to tensor<1024xf32>
    %9 = arith.extf %5 : tensor<1024xf16> to tensor<1024xf32>
    %10 = math.sqrt %3 : tensor<1024xf32>
    tt.store %arg3, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xbf16>
    tt.store %arg4, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg5, %8 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg6, %9 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg7, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }
}

