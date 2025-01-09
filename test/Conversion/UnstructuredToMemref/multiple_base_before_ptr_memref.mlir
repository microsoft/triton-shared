#map = affine_map<(d0) -> (d0)>
module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %2 = arith.addi %1, %cst : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tensor.empty() : tensor<64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>) outs(%4 : tensor<64xf32>) {
    ^bb0(%in: !tt.ptr<f32>, %in_1: i32, %out: f32):
      %9 = builtin.unrealized_conversion_cast %in : !tt.ptr<f32> to memref<*xf32>
      %cast = memref.cast %9 : memref<*xf32> to memref<?xf32>
      %10 = bufferization.to_tensor %cast restrict : memref<?xf32>
      %11 = arith.index_cast %in_1 : i32 to index
      %extracted = tensor.extract %10[%11] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<64xf32>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tensor.empty() : tensor<64xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%6, %5, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi32>) outs(%7 : tensor<64xf32>) {
    ^bb0(%in: !tt.ptr<f32>, %in_1: f32, %in_2: i32, %out: f32):
      %9 = builtin.unrealized_conversion_cast %in : !tt.ptr<f32> to memref<*xf32>
      %cast = memref.cast %9 : memref<*xf32> to memref<?xf32>
      %10 = arith.index_cast %in_2 : i32 to index
      memref.store %in_1, %cast[%10] : memref<?xf32>
      linalg.yield %in_1 : f32
    } -> tensor<64xf32>
    tt.return
  }
}