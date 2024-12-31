#map = affine_map<(d0) -> (d0)>
module {
  func.func @gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.fill ins(%c4_i32 : i32) outs(%0 : tensor<4xi32>) -> tensor<4xi32>
    %c64_i32 = arith.constant 64 : i32
    %2 = tensor.empty() : tensor<4xi32>
    %3 = linalg.fill ins(%c64_i32 : i32) outs(%2 : tensor<4xi32>) -> tensor<4xi32>
    %c3_i32 = arith.constant 3 : i32
    %4 = tensor.empty() : tensor<4xi32>
    %5 = linalg.fill ins(%c3_i32 : i32) outs(%4 : tensor<4xi32>) -> tensor<4xi32>
    %6 = tensor.empty() : tensor<4xi32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %9 = linalg.index 0 : index
      %10 = arith.index_cast %9 : index to i32
      linalg.yield %10 : i32
    } -> tensor<4xi32>
    %8:6 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %7, %arg10 = %c0, %arg11 = %c1, %arg12 = %7, %arg13 = %c0, %arg14 = %c1) -> (tensor<4xi32>, index, index, tensor<4xi32>, index, index)  : i32 {
      %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %5 : tensor<4xi32>, tensor<4xi32>) outs(%arg9 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %21 = arith.divsi %in, %in_0 : i32
        linalg.yield %21 : i32
      } -> tensor<4xi32>
      %10 = tensor.empty() : tensor<4xi32>
      %11 = linalg.fill ins(%arg8 : i32) outs(%10 : tensor<4xi32>) -> tensor<4xi32>
      %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %11 : tensor<4xi32>, tensor<4xi32>) outs(%9 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %21 = arith.addi %in, %in_0 : i32
        linalg.yield %21 : i32
      } -> tensor<4xi32>
      %13 = tensor.empty() : tensor<4xi1>
      %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %3 : tensor<4xi32>, tensor<4xi32>) outs(%13 : tensor<4xi1>) {
      ^bb0(%in: i32, %in_0: i32, %out: i1):
        %21 = arith.cmpi slt, %in, %in_0 : i32
        linalg.yield %21 : i1
      } -> tensor<4xi1>
      %15 = "tts.make_unstructured_tptr"(%arg0, %12) : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
      %16 = tt.load %15, %14 : tensor<4x!tt.ptr<f32>>
      %17 = "tts.make_unstructured_tptr"(%arg1, %12) : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
      tt.store %17, %16 : tensor<4x!tt.ptr<f32>>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %1 : tensor<4xi32>, tensor<4xi32>) outs(%12 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %21 = arith.addi %in, %in_0 : i32
        linalg.yield %21 : i32
      } -> tensor<4xi32>
      %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %1 : tensor<4xi32>, tensor<4xi32>) outs(%arg12 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %21 = arith.addi %in, %in_0 : i32
        linalg.yield %21 : i32
      } -> tensor<4xi32>
      %structured, %offsets, %strides = "tts.get_structured_state"(%18) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi32>) -> (tensor<4xi32>, index, index)
      %20 = arith.addi %arg13, %c4 : index
      scf.yield %18, %offsets, %strides, %19, %20, %arg14 : tensor<4xi32>, index, index, tensor<4xi32>, index, index
    }
    return
  }
}