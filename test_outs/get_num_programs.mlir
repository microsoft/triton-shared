module {
  tt.func public @num_programs(%arg0: !tt.ptr<i32, 1>) {
    %0 = tt.get_num_programs {axis = 0 : i32} : i32
    %1 = tt.get_num_programs {axis = 1 : i32} : i32
    %2 = tt.get_num_programs {axis = 2 : i32} : i32
    %3 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %4 = tt.splat %0 : (i32) -> tensor<1xi32>
    "tts.store"(%3, %4) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    %5 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [1], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %6 = tt.splat %1 : (i32) -> tensor<1xi32>
    "tts.store"(%5, %6) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    %7 = tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [2], shape: [0], order: [] : <i32, 1> to tensor<1x!tt.ptr<i32, 1>>
    %8 = tt.splat %2 : (i32) -> tensor<1xi32>
    "tts.store"(%7, %8) <{static_dims = array<i64>}> : (tensor<1x!tt.ptr<i32, 1>>, tensor<1xi32>) -> ()
    tt.return
  }
}

