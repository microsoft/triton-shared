module {
  func.func @kernel(%arg0: f32, %arg1: bf16, %arg2: tensor<1024x!tt.ptr<f32, 1>>, %arg3: tensor<128x256x!tt.ptr<bf16, 1>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = tensor.empty() : tensor<128x256xbf16>
    %3 = linalg.fill ins(%arg1 : bf16) outs(%2 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    tt.store %arg2, %1 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg3, %3 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
    return
  }
}

