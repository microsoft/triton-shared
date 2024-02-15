module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: tensor<32x16x!tt.ptr<bf16, 1>>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 256, 16], strides: [%c256, 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[?, 1, 1]>>
    %alloc = memref.alloc() : memref<32x256x16xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<32x256x16xbf16, strided<[?, 1, 1]>> to memref<32x256x16xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32x256x16xbf16>
    %1 = tensor.empty() : tensor<32x16xbf16>
    %2 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %reduced = linalg.reduce ins(%0 : tensor<32x256x16xbf16>) outs(%2 : tensor<32x16xbf16>) dimensions = [1] 
      (%in: bf16, %init: bf16) {
        %3 = arith.addf %in, %init : bf16
        linalg.yield %3 : bf16
      }
    tt.store %arg2, %reduced {cache = 1 : i32, evict = 1 : i32} : tensor<32x16xbf16>
    return
  }
}

