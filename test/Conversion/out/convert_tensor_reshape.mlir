#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @bcast_kernel_01(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c2048_i64 = arith.constant 2048 : i64
    %cst = arith.constant dense<[1, 32]> : tensor<2xi64>
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.muli %arg5, %c32_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<32xf32>
    %reshape = tensor.reshape %3(%cst) : (tensor<32xf32>, tensor<2xi64>) -> tensor<1x32xf32>
    %4 = tensor.empty() : tensor<64x32xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%reshape : tensor<1x32xf32>) outs(%4 : tensor<64x32xf32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x32xf32>
    %6 = tensor.empty() : tensor<1xi64>
    %7 = linalg.fill ins(%c2048_i64 : i64) outs(%6 : tensor<1xi64>) -> tensor<1xi64>
    %reshape_0 = tensor.reshape %5(%7) : (tensor<64x32xf32>, tensor<1xi64>) -> tensor<2048xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [2048], strides: [1] : memref<*xf32> to memref<2048xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %reshape_0 in writable %reinterpret_cast_1 : (tensor<2048xf32>, memref<2048xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

