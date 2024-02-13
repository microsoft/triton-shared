module {
  func.func @triton__0d1d2de(%arg0: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c4 = arith.constant 4 : index
    %c77_i32 = arith.constant 77 : i32
    %c128 = arith.constant 128 : index
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.muli %arg6, %c4_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %1, %c128 : index
    %3 = tts.make_tptr %arg0 to sizes: [4, 64], strides: [%c128, 1], offsets: [%2, 0], shape: [0, 0], order: [] : <i32, 1> to tensor<4x64x!tt.ptr<i32, 1>>
    %4 = "tts.load"(%3, %c77_i32) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_dims = array<i64: 4, 32>}> : (tensor<4x64x!tt.ptr<i32, 1>>, i32) -> tensor<4x64xi32>
    %5 = tts.make_tptr %arg1 to sizes: [4, 64], strides: [1, %c4], offsets: [0, 0], shape: [0, 0], order: [] : <i32, 1> to tensor<4x64x!tt.ptr<i32, 1>>
    "tts.store"(%5, %4) <{static_dims = array<i64>}> : (tensor<4x64x!tt.ptr<i32, 1>>, tensor<4x64xi32>) -> ()
    return
  }
}

