module {
  func.func @maxnumf(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %2[] : tensor<f32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %3 = arith.maxnumf %in, %init : f32
        linalg.yield %3 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}


// -----
module {
  func.func @minnumf(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %cst = arith.constant 0x7F800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %2[] : tensor<f32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %3 = arith.minnumf %in, %init : f32
        linalg.yield %3 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}

