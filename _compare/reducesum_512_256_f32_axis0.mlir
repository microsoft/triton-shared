module {
  func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512, 256], strides: [%c256, 1] : memref<*xf32> to memref<512x256xf32, strided<[?, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<512x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<512x256xf32, strided<[?, 1]>> to memref<512x256xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<512x256xf32>
    %1 = tensor.empty() : tensor<256xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%0 : tensor<512x256xf32>) outs(%2 : tensor<256xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<256xf32>, memref<256xf32, strided<[1]>>) -> ()
    return
  }
}

