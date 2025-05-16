module {
  tt.func public @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.muli %arg2, %c2_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.index_cast %4 : i32 to index
    %7:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c0, %arg6 = %c0) -> (index, index)  : i32 {
      %8 = arith.addi %arg5, %c1 : index
      %9 = arith.addi %arg6, %c1 : index
      %10:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %8, %arg9 = %9) -> (index, index)  : i32 {
        %15 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg9, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %16 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg8, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %17 = "tts.load"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%15, %17) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %18 = arith.addi %arg8, %c3 : index
        %19 = arith.addi %arg9, %c3 : index
        scf.yield %18, %19 : index, index
      }
      %11 = arith.addi %arg5, %6 : index
      %12 = arith.addi %11, %c1 : index
      %13 = arith.addi %arg6, %5 : index
      %14 = arith.addi %13, %c1 : index
      scf.yield %12, %14 : index, index
    }
    tt.return
  }
  tt.func public @nested2_use_loop_results(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.muli %arg3, %c4_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.index_cast %4 : i32 to index
    %7 = arith.index_cast %4 : i32 to index
    %8 = arith.index_cast %4 : i32 to index
    %9:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c0, %arg6 = %c0) -> (index, index)  : i32 {
      %10 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg6, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %11 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg5, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %12 = "tts.load"(%11) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
      "tts.store"(%10, %12) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
      %13 = arith.addi %arg5, %8 : index
      %14 = arith.addi %arg6, %7 : index
      %15:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %13, %arg9 = %14) -> (index, index)  : i32 {
        %16 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg9, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %17 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg8, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %18 = "tts.load"(%17) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%16, %18) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %19 = arith.addi %arg8, %6 : index
        %20 = arith.addi %arg9, %5 : index
        scf.yield %19, %20 : index, index
      }
      scf.yield %15#0, %15#1 : index, index
    }
    tt.return
  }

  tt.func public @nested_use_same_level_loop_result(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
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
    %7 = arith.muli %arg3, %c2_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    %10 = arith.index_cast %7 : i32 to index
    %11 = arith.index_cast %7 : i32 to index
    %12 = arith.index_cast %7 : i32 to index
    %13 = arith.muli %arg3, %c2_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15:3 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c0, %arg6 = %4, %arg7 = %c0) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
      %16 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %arg5) -> (index)  : i32 {
        %19 = arith.addi %arg9, %6 : index
        scf.yield %19 : index
      }
      %17:3 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %16, %arg10 = %arg6, %arg11 = %arg7) -> (index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
        %19 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%arg11, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %20 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%arg9, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %21 = "tts.load"(%20) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        %22 = arith.addi %arg9, %12 : index
        %23 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%0, %1], offsets: [%22, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %24 = "tts.load"(%23) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        "tts.store"(%19, %21) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %25 = arith.addi %arg11, %11 : index
        %26 = arith.addi %25, %10 : index
        %27 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%26, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        "tts.store"(%27, %24) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %28 = arith.addi %26, %9 : index
        %29 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%2, %3], offsets: [%28, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
        %30 = arith.addi %22, %8 : index
        scf.yield %30, %29, %28 : index, tensor<2x2x!tt.ptr<f32>>, index
      }
      %18 = arith.addi %17#0, %14 : index
      scf.yield %18, %17#1, %17#2 : index, tensor<2x2x!tt.ptr<f32>>, index
    }
    tt.return
  }
}

