#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xi32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<1024xi32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xi32, strided<[1]>> to memref<1024xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xi32>
    %1 = tensor.empty() : tensor<1024xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xi32>) outs(%1 : tensor<1024xf32>) {
    ^bb0(%in: i32, %out: f32):
      %3 = arith.bitcast %in : i32 to f32
      linalg.yield %3 : f32
    } -> tensor<1024xf32>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<1024xf32>, memref<1024xf32, strided<[1]>>) -> ()
    return
  }
}

