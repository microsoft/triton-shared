module {
  tt.func public @wrap_side_by_side_masked_loop_01234567(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.muli %0, %c2 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.muli %3, %c6 : index
    %5 = arith.muli %2, %3 : index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.muli %arg4, %c4_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.muli %arg5, %c4_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %1, %arg10 = %c0) -> (index, index)  : i32 {
      %13 = tts.make_tptr %arg1 to sizes: [4, 4], strides: [%6, %7], offsets: [%arg10, %c0], shape: [0, 0], order: [] : <f32, 1> to tensor<4x4x!tt.ptr<f32, 1>>
      %14 = tts.make_tptr %arg0 to sizes: [4, 4], strides: [%0, %3], offsets: [%arg9, %4], shape: [0, %5], order: [] : <f32, 1> to tensor<4x4x!tt.ptr<f32, 1>>
      %15 = "tts.load"(%14, %cst) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_dims = array<i64: 2, 4>}> : (tensor<4x4x!tt.ptr<f32, 1>>, f32) -> tensor<4x4xf32>
      "tts.store"(%13, %15) <{static_dims = array<i64>}> : (tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xf32>) -> ()
      %16 = arith.addi %arg9, %9 : index
      %17 = arith.addi %arg10, %11 : index
      scf.yield %16, %17 : index, index
    }
    tt.return
  }
}

