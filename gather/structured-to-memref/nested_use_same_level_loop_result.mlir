module {
      func.func @nested_use_same_level_loop_result(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%0, %1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    %3 = arith.muli %arg3, %c2_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5:3 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %c0, %arg12 = %2, %arg13 = %c0) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
      %6 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %arg11) -> (index)  : i32 {
        %9 = arith.addi %arg15, %4 : index
        scf.yield %9 : index
      }
      %7:3 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %6, %arg16 = %arg12, %arg17 = %arg13) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
        %9 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg17, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %10 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg15, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %11 = "tts.load"(%10) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        %12 = arith.addi %arg15, %4 : index
        %13 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%12, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %14 = "tts.load"(%13) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%9, %11) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %15 = arith.addi %arg17, %4 : index
        %16 = arith.addi %15, %4 : index
        %17 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%0, %1], offsets: [%16, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        "tts.store"(%17, %14) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %18 = arith.addi %16, %4 : index
        %19 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%0, %1], offsets: [%18, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %20 = arith.addi %12, %4 : index
        scf.yield %20, %19, %18 : index, tensor<2x2x!tt.ptr<f32>>, index
      }
      %8 = arith.addi %7#0, %4 : index
      scf.yield %8, %7#1, %7#2 : index, tensor<2x2x!tt.ptr<f32>>, index
    }
    return
  }
}