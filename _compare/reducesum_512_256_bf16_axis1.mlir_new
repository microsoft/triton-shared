module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512, 256], strides: [%c256, 1] : memref<*xbf16> to memref<512x256xbf16, strided<[?, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [1] : memref<*xbf16> to memref<512xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<512x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<512x256xbf16, strided<[?, 1]>> to memref<512x256xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<512x256xbf16>
    %1 = tensor.empty() : tensor<256x512xbf16>
    %transposed = linalg.transpose ins(%0 : tensor<512x256xbf16>) outs(%1 : tensor<256x512xbf16>) permutation = [1, 0] 
    %2 = tensor.empty() : tensor<512xbf16>
    %3 = linalg.fill ins(%cst : bf16) outs(%2 : tensor<512xbf16>) -> tensor<512xbf16>
    %reduced = linalg.reduce ins(%transposed : tensor<256x512xbf16>) outs(%3 : tensor<512xbf16>) dimensions = [0] 
      (%in: bf16, %init: bf16) {
        %4 = arith.addf %in, %init : bf16
        linalg.yield %4 : bf16
      }
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<512xbf16>, memref<512xbf16, strided<[1]>>) -> ()
    return
  }
}

