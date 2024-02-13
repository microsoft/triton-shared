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
    %4 = tt.addptr %arg1, %2 : !tt.ptr<f32, 1>, i32
    %5:3 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %4, %arg13 = %1, %arg14 = %3) -> (!tt.ptr<f32, 1>, tensor<1024xf32>, index) {
      %9 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%arg14], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
      %10 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>) -> tensor<1024xf32>
      %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = math.exp %in : f32
        linalg.yield %16 : f32
      } -> tensor<1024xf32>
      %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %11 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %16 = arith.addf %in, %in_0 : f32
        linalg.yield %16 : f32
      } -> tensor<1024xf32>
      %13 = arith.index_cast %arg11 : index to i32
      %14 = tt.addptr %arg12, %13 : !tt.ptr<f32, 1>, i32
      %15 = arith.addi %arg14, %arg11 : index
      scf.yield %14, %12, %15 : !tt.ptr<f32, 1>, tensor<1024xf32>, index
    }
    %6 = arith.muli %arg8, %arg3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%7], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    "tts.store"(%8, %5#1) <{static_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>) -> ()
    return
  }
}

