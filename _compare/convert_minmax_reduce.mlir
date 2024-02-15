module {
  func.func @minmax_sgt(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<4096xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4096xi32>) -> tensor<4096xi32>
    %2 = bufferization.alloc_tensor() : tensor<i32>
    %inserted = tensor.insert %c-2147483648_i32 into %2[] : tensor<i32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xi32>) outs(%inserted : tensor<i32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %3 = arith.maxsi %in, %init : i32
        linalg.yield %3 : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xi32, strided<[1]>>
    return
  }
}


// -----
module {
  func.func @minmax_ugt(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<4096xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4096xi32>) -> tensor<4096xi32>
    %2 = bufferization.alloc_tensor() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %2[] : tensor<i32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xi32>) outs(%inserted : tensor<i32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %3 = arith.maxui %in, %init : i32
        linalg.yield %3 : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xi32, strided<[1]>>
    return
  }
}


// -----
module {
  func.func @minmax_slt(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<4096xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4096xi32>) -> tensor<4096xi32>
    %2 = bufferization.alloc_tensor() : tensor<i32>
    %inserted = tensor.insert %c2147483647_i32 into %2[] : tensor<i32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xi32>) outs(%inserted : tensor<i32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %3 = arith.minsi %in, %init : i32
        linalg.yield %3 : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xi32, strided<[1]>>
    return
  }
}


// -----
module {
  func.func @minmax_ult(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<4096xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4096xi32>) -> tensor<4096xi32>
    %2 = bufferization.alloc_tensor() : tensor<i32>
    %inserted = tensor.insert %c-1_i32 into %2[] : tensor<i32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xi32>) outs(%inserted : tensor<i32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %3 = arith.minui %in, %init : i32
        linalg.yield %3 : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
    affine.store %extracted, %reinterpret_cast[0] : memref<1xi32, strided<[1]>>
    return
  }
}

