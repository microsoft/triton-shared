#map = affine_map<(d0) -> (d0)>
module {
  func.func @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tensor.empty() : tensor<64xi32>
    %1 = linalg.fill ins(%c64_i32 : i32) outs(%0 : tensor<64xi32>) -> tensor<64xi32>
    %c64_i32_0 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %c10_i32 = arith.constant 10 : i32
    %2 = tensor.empty() : tensor<64xi32>
    %3 = linalg.fill ins(%c10_i32 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
    %4 = tensor.empty() : tensor<64xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %7 = linalg.index 0 : index
      %8 = arith.index_cast %7 : index to i32
      linalg.yield %8 : i32
    } -> tensor<64xi32>
    %6:6 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %5, %arg10 = %c0, %arg11 = %c1, %arg12 = %5, %arg13 = %c0, %arg14 = %c1) -> (tensor<64xi32>, index, index, tensor<64xi32>, index, index)  : i32 {
      %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %3 : tensor<64xi32>, tensor<64xi32>) outs(%arg9 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %19 = arith.divsi %in, %in_1 : i32
        linalg.yield %19 : i32
      } -> tensor<64xi32>
      %8 = arith.addi %arg8, %c5_i32 : i32
      %9 = arith.remsi %8, %c64_i32_0 : i32
      %10 = tensor.empty() : tensor<64xi32>
      %11 = linalg.fill ins(%9 : i32) outs(%10 : tensor<64xi32>) -> tensor<64xi32>
      %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %11 : tensor<64xi32>, tensor<64xi32>) outs(%7 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %19 = arith.addi %in, %in_1 : i32
        linalg.yield %19 : i32
      } -> tensor<64xi32>
      %13 = "tts.create_ptr"(%arg0, %12) : (!tt.ptr<f32>, tensor<64xi32>) -> tensor<64x!tt.ptr<f32>>
      %14 = tt.load %13 : tensor<64x!tt.ptr<f32>>
      %15 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
      %reinterpret_cast = memref.reinterpret_cast %15 to offset: [%arg13], sizes: [64], strides: [%arg14] : memref<*xf32> to memref<64xf32, strided<[?], offset: ?>>
      bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64xf32>, memref<64xf32, strided<[?], offset: ?>>) -> ()
      %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %1 : tensor<64xi32>, tensor<64xi32>) outs(%12 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %19 = arith.addi %in, %in_1 : i32
        linalg.yield %19 : i32
      } -> tensor<64xi32>
      %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %1 : tensor<64xi32>, tensor<64xi32>) outs(%arg12 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_1: i32, %out: i32):
        %19 = arith.addi %in, %in_1 : i32
        linalg.yield %19 : i32
      } -> tensor<64xi32>
      %structured, %offsets, %strides = "tts.get_structured_state"(%16) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<64xi32>) -> (tensor<64xi32>, index, index)
      %18 = arith.addi %arg13, %c64 : index
      scf.yield %16, %offsets, %strides, %17, %18, %arg14 : tensor<64xi32>, index, index, tensor<64xi32>, index, index
    }
    return
  }
}