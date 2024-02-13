#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x256xbf16>) -> tensor<4x256xbf16>
    %c1 = arith.constant 1 : index
    %c256_i32 = arith.constant 256 : i32
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %2 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<1x256x!tt.ptr<bf16, 1>>
    %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16, 1>>) -> tensor<1x256xbf16>
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [%4, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<1x256x!tt.ptr<bf16, 1>>
    "tts.store"(%5, %3) <{static_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16, 1>>, tensor<1x256xbf16>) -> ()
    %6:2 = scf.for %arg8 = %c0 to %c12 step %c3 iter_args(%arg9 = %1, %arg10 = %c0) -> (tensor<4x256xbf16>, index) {
      %8 = arith.index_cast %arg8 : index to i32
      %9 = arith.muli %8, %c256_i32 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%10, %c1], offsets: [%arg10, %c0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
      %12 = "tts.load"(%11) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg9, %12 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%arg9 : tensor<4x256xbf16>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
        %15 = arith.addf %in, %in_0 : bf16
        linalg.yield %15 : bf16
      } -> tensor<4x256xbf16>
      %14 = arith.addi %arg10, %c256 : index
      scf.yield %13, %14 : tensor<4x256xbf16>, index
    }
    %7 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    "tts.store"(%7, %6#0) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    return
  }
}

