#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = arith.muli %arg8, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4:2 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %3) -> (tensor<1024xf32>, index) {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%arg13], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xf32>
      memref.copy %reinterpret_cast_0, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
      %7 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%7 : tensor<1024xf32>) outs(%7 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = math.exp %in : f32
        linalg.yield %11 : f32
      } -> tensor<1024xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %8 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg12 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %11 = arith.addf %in, %in_1 : f32
        linalg.yield %11 : f32
      } -> tensor<1024xf32>
      %10 = arith.addi %arg13, %arg11 : index
      scf.yield %9, %10 : tensor<1024xf32>, index
    }
    %5 = arith.muli %arg8, %arg3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %4#0 in writable %reinterpret_cast : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

