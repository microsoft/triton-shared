module {
  tt.func public @bcast_kernel_01(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>) -> tensor<32xf32>
    %6 = tt.reshape %5 {allow_reorder = false} : tensor<32xf32> -> tensor<1x32xf32>
    %7 = tt.broadcast %6 : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %8 = tt.reshape %7 {allow_reorder = false} : tensor<64x32xf32> -> tensor<2048xf32>
    %9 = tts.make_tptr %arg1 to sizes: [2048], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<2048x!tt.ptr<f32, 1>>
    "tts.store"(%9, %8) <{static_dims = array<i64>}> : (tensor<2048x!tt.ptr<f32, 1>>, tensor<2048xf32>) -> ()
    tt.return
  }
}

