#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_cumsum_op_012(%arg0: memref<*xf32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.muli %arg6, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<4096xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<4096xf32>
    %4 = tensor.empty() : tensor<4096xf32>
    %5 = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins(%3 : tensor<4096xf32>) outs(%4 : tensor<4096xf32>) -> tensor<4096xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [4096], strides: [1] : memref<*xi32> to memref<4096xi32, strided<[1], offset: ?>>
    %6 = tensor.empty() : tensor<4096xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<4096xf32>) outs(%6 : tensor<4096xi32>) {
    ^bb0(%in: f32, %out: i32):
      %8 = arith.fptosi %in : f32 to i32
      linalg.yield %8 : i32
    } -> tensor<4096xi32>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_0 : (tensor<4096xi32>, memref<4096xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}

