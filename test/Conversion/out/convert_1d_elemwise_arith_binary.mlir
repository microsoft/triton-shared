#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: tensor<1024x!tt.ptr<f32, 1>>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<1024xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1]>> to memref<1024xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %alloc_1 = memref.alloc() : memref<1024xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xf32, strided<[1]>> to memref<1024xf32>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.addf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1024xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%2, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.subf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%3 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1024xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.divf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1024xf32>
    %6 = tensor.empty() : tensor<1024xi1>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xi1>) {
    ^bb0(%in: f32, %in_2: f32, %out: i1):
      %9 = arith.cmpf oeq, %in, %in_2 : f32
      linalg.yield %9 : i1
    } -> tensor<1024xi1>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%7, %0, %1 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
    ^bb0(%in: i1, %in_2: f32, %in_3: f32, %out: f32):
      %9 = arith.select %in, %in_2, %in_3 : f32
      linalg.yield %9 : f32
    } -> tensor<1024xf32>
    tt.store %arg2, %8 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    return
  }
}

