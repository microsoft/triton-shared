#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<128x128xf32>, %arg3: memref<128x128xf32>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = builtin.unrealized_conversion_cast %arg3 : memref<128x128xf32> to tensor<128x128x!tt.ptr<f32>>
    %1 = builtin.unrealized_conversion_cast %arg2 : memref<128x128xf32> to tensor<128x128x!tt.ptr<f32>>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
    %alloc = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
    %alloc_1 = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = arith.addf %in, %in_2 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = arith.subf %in, %in_2 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %6 = "tts.make_unstructured_tptr"(%1, %c0_i32) : (tensor<128x128x!tt.ptr<f32>>, i32) -> tensor<128x128x!tt.ptr<f32>>
    tt.store %6, %4 : tensor<128x128x!tt.ptr<f32>>
    %7 = "tts.make_unstructured_tptr"(%0, %c0_i32) : (tensor<128x128x!tt.ptr<f32>>, i32) -> tensor<128x128x!tt.ptr<f32>>
    tt.store %7, %5 : tensor<128x128x!tt.ptr<f32>>
    return
  }
}
