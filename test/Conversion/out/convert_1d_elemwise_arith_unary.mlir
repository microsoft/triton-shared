#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xi32>, %arg2: memref<*xf16>, %arg3: tensor<1024x!tt.ptr<bf16, 1>>, %arg4: tensor<1024x!tt.ptr<f32, 1>>, %arg5: tensor<1024x!tt.ptr<f32, 1>>, %arg6: tensor<1024x!tt.ptr<f32, 1>>, %arg7: tensor<1024x!tt.ptr<f32, 1>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<*xf16> to memref<1024xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<1024xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1]>> to memref<1024xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %alloc_2 = memref.alloc() : memref<1024xi32>
    memref.copy %reinterpret_cast_0, %alloc_2 : memref<1024xi32, strided<[1]>> to memref<1024xi32>
    %1 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xi32>
    %alloc_3 = memref.alloc() : memref<1024xf16>
    memref.copy %reinterpret_cast_1, %alloc_3 : memref<1024xf16, strided<[1]>> to memref<1024xf16>
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<1024xf16>
    %3 = tensor.empty() : tensor<1024xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf32>) outs(%3 : tensor<1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %11 = arith.truncf %in : f32 to bf16
      linalg.yield %11 : bf16
    } -> tensor<1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = math.exp %in : f32
      linalg.yield %11 : f32
    } -> tensor<1024xf32>
    %6 = tensor.empty() : tensor<1024xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<1024xi32>) outs(%6 : tensor<1024xf32>) {
    ^bb0(%in: i32, %out: f32):
      %11 = arith.sitofp %in : i32 to f32
      linalg.yield %11 : f32
    } -> tensor<1024xf32>
    %8 = tensor.empty() : tensor<1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<1024xf16>) outs(%8 : tensor<1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %11 = arith.extf %in : f16 to f32
      linalg.yield %11 : f32
    } -> tensor<1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = math.sqrt %in : f32
      linalg.yield %11 : f32
    } -> tensor<1024xf32>
    tt.store %arg3, %4 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xbf16>
    tt.store %arg4, %5 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg5, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg6, %9 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg7, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    return
  }
}

