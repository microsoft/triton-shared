#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xi32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c6144 = arith.constant 6144 : index
    %c6 = arith.constant 6 : index
    %0 = tensor.empty() : tensor<256xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<256xi32>) {
    ^bb0(%out: i32):
      %7 = linalg.index 0 : index
      %8 = arith.index_cast %7 : index to i32
      linalg.yield %8 : i32
    } -> tensor<256xi32>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<256xi32> into tensor<256x1xi32>
    %2 = tensor.empty() : tensor<256x128xi32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<256x1xi32>) outs(%2 : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<256x128xi32>
    %c512 = arith.constant 512 : index
    %4 = arith.addi %c512, %c6144 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [256, 128], strides: [1, %c6] : memref<*xbf16> to memref<256x128xbf16, strided<[1, ?], offset: 512>>
    %alloc = memref.alloc() : memref<256x128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<256x128xbf16, strided<[1, ?], offset: 512>> to memref<256x128xbf16>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 512>>) -> ()
    %c512_0 = arith.constant 512 : index
    %6 = arith.addi %c512_0, %c6144 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [256, 128], strides: [1, %c6] : memref<*xi32> to memref<256x128xi32, strided<[1, ?], offset: 512>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_1 : (tensor<256x128xi32>, memref<256x128xi32, strided<[1, ?], offset: 512>>) -> ()
    return
  }
}

