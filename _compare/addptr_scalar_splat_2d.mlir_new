#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c128 = arith.constant 128 : index
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.addi %1, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x128xf32, strided<[1, 1], offset: ?>> to memref<128x128xf32>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.exp %in : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %5 = arith.muli %arg8, %arg3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.addi %6, %c128 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<128x128xf32>, memref<128x128xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

