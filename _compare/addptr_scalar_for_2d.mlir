#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = arith.muli %arg8, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4:2 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %3) -> (tensor<128x128xf32>, index) {
      %8 = arith.addi %arg13, %c128 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%8], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf32>
      memref.copy %reinterpret_cast_0, %alloc : memref<128x128xf32, strided<[1, 1], offset: ?>> to memref<128x128xf32>
      %9 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
      %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<128x128xf32>) outs(%9 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = math.exp %in : f32
        linalg.yield %13 : f32
      } -> tensor<128x128xf32>
      %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %10 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg12 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %13 = arith.addf %in, %in_1 : f32
        linalg.yield %13 : f32
      } -> tensor<128x128xf32>
      %12 = arith.addi %arg13, %arg11 : index
      scf.yield %11, %12 : tensor<128x128xf32>, index
    }
    %5 = arith.muli %arg8, %arg3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.addi %6, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %4#0 in writable %reinterpret_cast : (tensor<128x128xf32>, memref<128x128xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

