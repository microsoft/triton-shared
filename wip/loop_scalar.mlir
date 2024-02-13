#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %2 = arith.muli %arg8, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = tt.addptr %arg1, %2 : !tt.ptr<f32, 1>, i32
    %5:3 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %4, %arg14 = %3) -> (tensor<128x128xf32>, !tt.ptr<f32, 1>, index) {
      %10 = arith.addi %arg14, %c128 : index
      %11 = tts.make_tptr %arg1 to sizes: [128, 128], strides: [1, 1], offsets: [%10, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
      %12 = "tts.load"(%11) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
      %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<128x128xf32>) outs(%12 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %18 = math.exp %in : f32
        linalg.yield %18 : f32
      } -> tensor<128x128xf32>
      %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %13 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg12 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %18 = arith.addf %in, %in_0 : f32
        linalg.yield %18 : f32
      } -> tensor<128x128xf32>
      %15 = arith.index_cast %arg11 : index to i32
      %16 = tt.addptr %arg13, %15 : !tt.ptr<f32, 1>, i32
      %17 = arith.addi %arg14, %arg11 : index
      scf.yield %14, %16, %17 : tensor<128x128xf32>, !tt.ptr<f32, 1>, index
    }
    %6 = arith.muli %arg8, %arg3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.addi %7, %c128 : index
    %9 = tts.make_tptr %arg0 to sizes: [128, 128], strides: [1, 1], offsets: [%8, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    "tts.store"(%9, %5#0) <{static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>, tensor<128x128xf32>) -> ()
    return
  }
}

