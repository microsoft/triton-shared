module {
  tt.func public @nested1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    %2 = arith.index_cast %arg6 : i32 to index
    %3 = arith.index_cast %arg7 : i32 to index
    %4 = arith.muli %arg5, %c32_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %c0, %arg10 = %c0) -> (index, index)  : i32 {
      %7 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg10, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %8 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg9, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %9 = "tts.load"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
      "tts.store"(%7, %9) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
      %10 = arith.addi %arg9, %5 : index
      %11 = arith.addi %arg10, %5 : index
      scf.yield %10, %11 : index, index
    }
    tt.return
  }
}

