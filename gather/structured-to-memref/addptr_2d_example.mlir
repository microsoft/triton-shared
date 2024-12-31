#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c5 = arith.constant 5 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c5], offsets: [%0, 0], shape: [0, 0], order: [] : !tt.ptr<bf16> to tensor<4x256x!tt.ptr<bf16>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = tts.make_tptr %arg1 to sizes: [4, 256], strides: [1, %c5], offsets: [%3, 0], shape: [0, 0], order: [] : !tt.ptr<bf16> to tensor<4x256x!tt.ptr<bf16>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %5 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%2 : tensor<4x256xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %9 = arith.addf %in, %in_0 : bf16
      linalg.yield %9 : bf16
    } -> tensor<4x256xbf16>
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = tts.make_tptr %arg2 to sizes: [4, 256], strides: [1, %c5], offsets: [%7, 0], shape: [0, 0], order: [] : !tt.ptr<bf16> to tensor<4x256x!tt.ptr<bf16>>
    "tts.store"(%8, %6) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
    return
  }
}