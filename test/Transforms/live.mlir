  tt.func public @nested3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    %5 = arith.muli %arg3, %c2_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    %8 = arith.index_cast %5 : i32 to index
    %9 = arith.index_cast %5 : i32 to index
    %10 = arith.index_cast %5 : i32 to index
    %11 = arith.muli %arg3, %c2_i32 : i32
    %12 = arith.index_cast %11 : i32 to index
    %13:3 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c0, %arg6 = %4, %arg7 = %c0) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
      %14 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg5, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %15 = "tts.load"(%14) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
      %16:3 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
        %18 = arith.addi %arg9, %10 : index
        %19 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%18, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %20 = "tts.load"(%19) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        %21:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %18, %arg14 = %arg10, %arg15 = %arg11) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
          %22 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg15, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
          %23 = arith.addi %arg13, %9 : index
          %24 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%23, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
          %25 = "tts.load"(%24) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
          "tts.store"(%22, %15) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
          %26 = arith.addi %arg15, %8 : index
          %27 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%26, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
          "tts.store"(%27, %20) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
          %28 = arith.addi %26, %7 : index
          %29 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%28, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
          "tts.store"(%29, %25) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
          %30 = arith.addi %28, %6 : index
          %31 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%30, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
          scf.yield %23, %31, %30 : index, tensor<2x2x!tt.ptr<f32>>, index
        }
        scf.yield %21#0, %21#1, %21#2 : index, tensor<2x2x!tt.ptr<f32>>, index
      }
      %17 = arith.addi %16#0, %12 : index
      scf.yield %17, %16#1, %16#2 : index, tensor<2x2x!tt.ptr<f32>>, index
    }
    tt.return
  }