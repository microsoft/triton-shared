module {
  func.func @kernel(%arg0: f32, %arg1: bf16, %arg2: memref<1024xf32>, %arg3: memref<128x256xbf16>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = tensor.empty() : tensor<128x256xbf16>
    %3 = linalg.fill ins(%arg1 : bf16) outs(%2 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    bufferization.materialize_in_destination %1 in writable %arg2 : (tensor<1024xf32>, memref<1024xf32>) -> ()
    bufferization.materialize_in_destination %3 in writable %arg3 : (tensor<128x256xbf16>, memref<128x256xbf16>) -> ()
    return
  }
}

