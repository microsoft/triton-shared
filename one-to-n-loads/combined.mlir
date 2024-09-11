module {
  tt.func public @one_to_n_loads_simplified(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tts.make_tptr %arg1 to sizes: [2], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<2x!tt.ptr<i32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x!tt.ptr<i32>>) -> tensor<2xi32>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %4 = arith.muli %2, %3 : tensor<2x1xi32>
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = tts.make_gather_tptr offsets: %4, %arg0 to sizes: [1, 4], strides: [0, %5] : tensor<2x1xi32> <f32> to tensor<2x4x!tt.ptr<f32>>
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.index_cast %arg4 : i32 to index
    %9 = tts.make_tptr %arg2 to sizes: [2, 4], strides: [%7, %8], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
    %10 = arith.muli %arg4, %c4_i32 : i32
    %11 = tt.splat %10 : i32 -> tensor<2x4xi32>
    %12:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %6, %arg7 = %9) -> (tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>)  : i32 {
      %13 = tt.load %arg6 : tensor<2x4x!tt.ptr<f32>>
      tt.store %arg7, %13 : tensor<2x4x!tt.ptr<f32>>
      %14 = tt.addptr %arg6, %11 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %15 = tt.addptr %arg7, %11 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %14, %15 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}