#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel_0d1d2de3de4de5c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x2xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x2xf32>) -> tensor<4x2xf32>
    %2 = tensor.empty() : tensor<4x2xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2xf32>) -> tensor<4x2xf32>
    %4 = arith.muli %arg8, %c4_i32 : i32
    %5 = arith.muli %arg9, %c2_i32 : i32
    %6 = arith.index_cast %4 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    %8 = arith.index_cast %arg4 : i32 to index
    %9 = arith.muli %6, %8 : index
    %10 = arith.addi %9, %7 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [4, 2], strides: [%8, %c1] : memref<*xf32> to memref<4x2xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x2xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x2xf32, strided<[?, ?], offset: ?>> to memref<4x2xf32>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<4x2xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %3 : tensor<4x2xf32>, tensor<4x2xf32>) outs(%11 : tensor<4x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %19 = arith.mulf %in, %in_2 : f32
      linalg.yield %19 : f32
    } -> tensor<4x2xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %1 : tensor<4x2xf32>, tensor<4x2xf32>) outs(%12 : tensor<4x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %19 = arith.addf %in, %in_2 : f32
      linalg.yield %19 : f32
    } -> tensor<4x2xf32>
    %14 = arith.index_cast %4 : i32 to index
    %15 = arith.index_cast %5 : i32 to index
    %16 = arith.index_cast %arg4 : i32 to index
    %17 = arith.muli %14, %16 : index
    %18 = arith.addi %17, %15 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [4, 2], strides: [%16, %c1] : memref<*xf32> to memref<4x2xf32, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %13 in writable %reinterpret_cast_1 : (tensor<4x2xf32>, memref<4x2xf32, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

