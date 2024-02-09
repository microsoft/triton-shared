module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128, 64], strides: [%c128, 1] : memref<*xbf16> to memref<128x64xbf16, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<128x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<128x64xbf16, strided<[?, 1]>> to memref<128x64xbf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 256], strides: [%c256, 1] : memref<*xbf16> to memref<64x256xbf16, strided<[?, 1]>>
    %alloc_1 = memref.alloc() : memref<64x256xbf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x256xbf16, strided<[?, 1]>> to memref<64x256xbf16>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x256xbf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 256], strides: [%c256, 1] : memref<*xbf16> to memref<128x256xbf16, strided<[?, 1]>>
    %4 = tensor.empty() : tensor<128x256xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%4 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %6 = linalg.matmul ins(%2, %3 : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%5 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<128x256xbf16>, memref<128x256xbf16, strided<[?, 1]>>) -> ()
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast_2 : (tensor<128x256xbf16>, memref<128x256xbf16, strided<[?, 1]>>) -> ()
    return
  }
}

