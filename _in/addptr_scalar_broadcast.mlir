#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024, 1024], strides: [1, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[1, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1024x1024xf32, strided<[1, 1], offset: ?>> to memref<1024x1024xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1024x1024xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.exp %in : f32
      linalg.yield %6 : f32
    } -> tensor<1024x1024xf32>
    %4 = arith.muli %arg8, %arg3 : i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [1024, 1024], strides: [1, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<1024x1024xf32>, memref<1024x1024xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

