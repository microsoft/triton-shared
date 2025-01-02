#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: tensor<1024x!tt.ptr<f32>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %0 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i1> to tensor<1024x!tt.ptr<i1>>
    %1 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %2 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %3 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<i1>>) -> tensor<1024xi1>
    %4 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %5 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4, %5 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %7 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1024xf32>
    tt.store %arg3, %6 : tensor<1024x!tt.ptr<f32>>
    return
  }
}
