#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %2 = arith.muli %arg8, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4:2 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %3) -> (tensor<1024xf32>, index) {
      %8 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%arg13], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
      %9 = "tts.load"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
      %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%9 : tensor<1024xf32>) outs(%9 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = math.exp %in : f32
        linalg.yield %13 : f32
      } -> tensor<1024xf32>
      %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %10 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg12 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %13 = arith.addf %in, %in_0 : f32
        linalg.yield %13 : f32
      } -> tensor<1024xf32>
      %12 = arith.addi %arg13, %arg11 : index
      scf.yield %11, %12 : tensor<1024xf32>, index
    }
    %5 = arith.muli %arg8, %arg3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%6], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    "tts.store"(%7, %4#0) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>) -> ()
    return
  }
}

