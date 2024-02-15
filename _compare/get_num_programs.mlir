module {
  func.func @num_programs(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %0 = tensor.empty() : tensor<1xi32>
    %1 = linalg.fill ins(%arg1 : i32) outs(%0 : tensor<1xi32>) -> tensor<1xi32>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%c1], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %2 = tensor.empty() : tensor<1xi32>
    %3 = linalg.fill ins(%arg2 : i32) outs(%2 : tensor<1xi32>) -> tensor<1xi32>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%c2], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %4 = tensor.empty() : tensor<1xi32>
    %5 = linalg.fill ins(%arg3 : i32) outs(%4 : tensor<1xi32>) -> tensor<1xi32>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_1 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}

