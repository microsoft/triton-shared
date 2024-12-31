module {
  func.func @nested2_use_loop_results(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [0, 0], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
    %3 = arith.index_cast %arg2 : i32 to index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%3, %4], offsets: [0, 0], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
    %6 = arith.muli %arg3, %c4_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    %9 = arith.index_cast %6 : i32 to index
    %10 = arith.index_cast %6 : i32 to index
    %11:10 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %2, %arg12 = %c0, %arg13 = %c0, %arg14 = %0, %arg15 = %1, %arg16 = %5, %arg17 = %c0, %arg18 = %c0, %arg19 = %3, %arg20 = %4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
      %12 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg19, %arg20], offsets: [%arg17, %arg18], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
      %13 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg14, %arg15], offsets: [%arg12, %arg13], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
      %14 = "tts.load"(%13) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
      "tts.store"(%12, %14) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
      %15 = arith.addi %arg12, %10 : index
      %16 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg14, %arg15], offsets: [%15, %arg13], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
      %17 = arith.addi %arg17, %9 : index
      %18 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg19, %arg20], offsets: [%17, %arg18], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
      %19:10 = scf.for %arg21 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg22 = %16, %arg23 = %15, %arg24 = %arg13, %arg25 = %arg14, %arg26 = %arg15, %arg27 = %18, %arg28 = %17, %arg29 = %arg18, %arg30 = %arg19, %arg31 = %arg20) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
        %20 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg30, %arg31], offsets: [%arg28, %arg29], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
        %21 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg25, %arg26], offsets: [%arg23, %arg24], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
        %22 = "tts.load"(%21) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%20, %22) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %23 = arith.addi %arg23, %8 : index
        %24 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%arg25, %arg26], offsets: [%23, %arg24], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
        %25 = arith.addi %arg28, %7 : index
        %26 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%arg30, %arg31], offsets: [%25, %arg29], shape: [0, 0], order: [] : !tt.ptr<f32> to tensor<2x2x!tt.ptr<f32>>
        scf.yield %24, %23, %arg24, %arg25, %arg26, %26, %25, %arg29, %arg30, %arg31 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
      }
      scf.yield %19#0, %19#1, %19#2, %19#3, %19#4, %19#5, %19#6, %19#7, %19#8, %19#9 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
    }
    return
  }
}

