module {
  func.func @num_programs(%arg0: !tt.ptr<i32, 1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %0 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %1 = tensor.empty() : tensor<1xi32>
    %2 = linalg.fill ins(%arg1 : i32) outs(%1 : tensor<1xi32>) -> tensor<1xi32>
    "tts.store"(%0, %2) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    %3 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [1], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %4 = tensor.empty() : tensor<1xi32>
    %5 = linalg.fill ins(%arg2 : i32) outs(%4 : tensor<1xi32>) -> tensor<1xi32>
    "tts.store"(%3, %5) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    %6 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [2], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %7 = tensor.empty() : tensor<1xi32>
    %8 = linalg.fill ins(%arg3 : i32) outs(%7 : tensor<1xi32>) -> tensor<1xi32>
    "tts.store"(%6, %8) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    return
  }
}
