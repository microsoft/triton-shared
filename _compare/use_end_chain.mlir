#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c6 = arith.constant 6 : index
    %c6_i32 = arith.constant 6 : i32
    %0 = tensor.empty() : tensor<256x128xi32>
    %1 = linalg.fill ins(%c6_i32 : i32) outs(%0 : tensor<256x128xi32>) -> tensor<256x128xi32>
    %2 = tensor.empty() : tensor<256xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<256xi32>) {
    ^bb0(%out: i32):
      %15 = linalg.index 0 : index
      %16 = arith.index_cast %15 : index to i32
      linalg.yield %16 : i32
    } -> tensor<256xi32>
    %expanded = tensor.expand_shape %3 [[0, 1]] : tensor<256xi32> into tensor<256x1xi32>
    %4 = tensor.empty() : tensor<256x128xi32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<256x1xi32>) outs(%4 : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<256x128xi32>
    %6 = tensor.empty() : tensor<128xi32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<128xi32>) {
    ^bb0(%out: i32):
      %15 = linalg.index 0 : index
      %16 = arith.index_cast %15 : index to i32
      linalg.yield %16 : i32
    } -> tensor<128xi32>
    %expanded_0 = tensor.expand_shape %7 [[0, 1]] : tensor<128xi32> into tensor<1x128xi32>
    %8 = tensor.empty() : tensor<256x128xi32>
    %9 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded_0 : tensor<1x128xi32>) outs(%8 : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<256x128xi32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9, %1 : tensor<256x128xi32>, tensor<256x128xi32>) outs(%9 : tensor<256x128xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %15 = arith.muli %in, %in_1 : i32
      linalg.yield %15 : i32
    } -> tensor<256x128xi32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5, %10 : tensor<256x128xi32>, tensor<256x128xi32>) outs(%5 : tensor<256x128xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %15 = arith.addi %in, %in_1 : i32
      linalg.yield %15 : i32
    } -> tensor<256x128xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [6656], sizes: [256, 128], strides: [1, %c6] : memref<*xbf16> to memref<256x128xbf16, strided<[1, ?], offset: 6656>>
    %alloc = memref.alloc() : memref<256x128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<256x128xbf16, strided<[1, ?], offset: 6656>> to memref<256x128xbf16>
    %12 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
    bufferization.materialize_in_destination %12 in writable %reinterpret_cast : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 6656>>) -> ()
    %13 = tensor.empty() : tensor<256x128xbf16>
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<256x128xi32>) outs(%13 : tensor<256x128xbf16>) {
    ^bb0(%in: i32, %out: bf16):
      %15 = arith.sitofp %in : i32 to bf16
      linalg.yield %15 : bf16
    } -> tensor<256x128xbf16>
    bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 6656>>) -> ()
    return
  }
}

