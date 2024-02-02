module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%4], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %6 = arith.index_cast %1 : i32 to index
    %7 = arith.addi %6, %c1024 : index
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = arith.minsi %7, %8 : index
    %10 = arith.subi %9, %6 : index
    %11 = "tts.load"(%5, %10) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, index) -> tensor<1024xf32>
    %12 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %13 = arith.index_cast %1 : i32 to index
    %14 = arith.addi %13, %c1024 : index
    %15 = arith.index_cast %arg3 : i32 to index
    %16 = arith.minsi %14, %15 : index
    %17 = arith.subi %16, %13 : index
    %18 = "tts.load"(%12, %17) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, index) -> tensor<1024xf32>
    %19 = arith.addf %11, %18 : tensor<1024xf32>
    %20 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %21 = arith.index_cast %1 : i32 to index
    %22 = arith.addi %21, %c1024 : index
    %23 = arith.index_cast %arg3 : i32 to index
    %24 = arith.minsi %22, %23 : index
    %25 = arith.subi %24, %21 : index
    "tts.store"(%20, %19, %25) <{static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>, index) -> ()
    tt.return
  }
}

