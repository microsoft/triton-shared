#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %arg8, %arg3 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.addi %1, %3 : index
    %5 = arith.muli %arg8, %arg4 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.addi %4, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%8 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<1024xf32>
    %10 = arith.muli %arg8, %arg3 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_0 : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

