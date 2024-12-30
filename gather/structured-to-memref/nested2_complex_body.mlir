module {
  func.func @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    %3 = arith.index_cast %arg2 : i32 to index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%3, %4], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    %6 = arith.muli %arg2, %c2_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    %9:10 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %2, %arg12 = %c0, %arg13 = %c0, %arg14 = %0, %arg15 = %1, %arg16 = %5, %arg17 = %c0, %arg18 = %c0, %arg19 = %3, %arg20 = %4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
      %10 = arith.addi %arg12, %c1 : index
      %11 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg14, %arg15], offsets: [%10, %arg13], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %12 = arith.addi %arg17, %c1 : index
      %13 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg19, %arg20], offsets: [%12, %arg18], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %14:10 = scf.for %arg21 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg22 = %11, %arg23 = %10, %arg24 = %arg13, %arg25 = %arg14, %arg26 = %arg15, %arg27 = %13, %arg28 = %12, %arg29 = %arg18, %arg30 = %arg19, %arg31 = %arg20) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
        %21 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg30, %arg31], offsets: [%arg28, %arg29], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %22 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg25, %arg26], offsets: [%arg23, %arg24], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %23 = "tts.load"(%22) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%21, %23) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %24 = arith.addi %arg23, %c3 : index
        %25 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg25, %arg26], offsets: [%24, %arg24], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %26 = arith.addi %arg28, %c3 : index
        %27 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg30, %arg31], offsets: [%26, %arg29], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        scf.yield %25, %24, %arg24, %arg25, %arg26, %27, %26, %arg29, %arg30, %arg31 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
      }
      %15 = arith.addi %arg12, %8 : index
      %16 = arith.addi %15, %c1 : index
      %17 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg14, %arg15], offsets: [%16, %arg13], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %18 = arith.addi %arg17, %7 : index
      %19 = arith.addi %18, %c1 : index
      %20 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg19, %arg20], offsets: [%19, %arg18], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      scf.yield %17, %16, %arg13, %arg14, %arg15, %20, %19, %arg18, %arg19, %arg20 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
    }
    return
  }
}

