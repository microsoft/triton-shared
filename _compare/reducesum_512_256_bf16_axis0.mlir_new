module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512, 256], strides: [%c256, 1] : memref<*xbf16> to memref<512x256xbf16, strided<[?, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<512x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<512x256xbf16, strided<[?, 1]>> to memref<512x256xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<512x256xbf16>
    %1 = tensor.empty() : tensor<256xbf16>
    %2 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<256xbf16>) -> tensor<256xbf16>
    %reduced = linalg.reduce ins(%0 : tensor<512x256xbf16>) outs(%2 : tensor<256xbf16>) dimensions = [0] 
      (%in: bf16, %init: bf16) {
        %3 = arith.addf %in, %init : bf16
        linalg.yield %3 : bf16
      }
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<256xbf16>, memref<256xbf16, strided<[1]>>) -> ()
    return
  }
}

