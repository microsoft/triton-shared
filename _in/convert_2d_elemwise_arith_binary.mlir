#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: tensor<128x128x!tt.ptr<f32, 1>>, %arg3: tensor<128x128x!tt.ptr<f32, 1>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
    %alloc = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
    %alloc_1 = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.addf %in, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.subf %in, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    tt.store %arg2, %2 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.store %arg3, %3 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    return
  }
}

