module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<128xbf16, strided<[1]>> to memref<128xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
    %1 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %1[] : tensor<f32>
    %reduced = linalg.reduce ins(%0 : tensor<128xbf16>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: bf16, %init: f32) {
        %3 = arith.extf %in : bf16 to f32
        %4 = arith.addf %3, %init : f32
        linalg.yield %4 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %2 = arith.truncf %extracted : f32 to bf16
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16>
    affine.store %2, %reinterpret_cast_0[0] : memref<1xbf16>
    return
  }
}

