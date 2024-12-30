module {
  func.func @wrap_stacked_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %1, %c2 : index
    %3 = arith.muli %0, %1 : index
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = arith.muli %4, %c3 : index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.muli %arg5, %c4_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10:2 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %2, %arg16 = %c0) -> (index, index)  : i32 {
      %11 = tts.make_tptr %arg1 to sizes: [4, 4], strides: [%6, %7], offsets: [%arg16, %c0], shape: [0, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>
      %12 = tts.make_tptr %arg0 to sizes: [4, 4], strides: [%1, %4], offsets: [%arg15, %5], shape: [%3, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>
      %13 = "tts.load"(%12, %cst) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 4, 3>}> : (tensor<4x4x!tt.ptr<f32>>, f32) -> tensor<4x4xf32>
      "tts.store"(%11, %13) <{static_mask_dims = array<i64>}> : (tensor<4x4x!tt.ptr<f32>>, tensor<4x4xf32>) -> ()
      %14 = arith.addi %arg15, %9 : index
      %15 = arith.addi %arg16, %9 : index
      scf.yield %14, %15 : index, index
    }
    return
  }
}