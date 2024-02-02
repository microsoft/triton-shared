module {
  tt.func public @rand(%arg0: !tt.ptr<i32, 1>, %arg1: !tt.ptr<i32, 1>) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tts.make_tptr %arg0 to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<8x!tt.ptr<i32, 1>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<8x!tt.ptr<i32, 1>>) -> tensor<8xi32>
    %3 = tt.extern_elementwise %2, %0 {libname = "libdevice", libpath = "/path/to/something", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %4 = tts.make_tptr %arg1 to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i32, 1> to tensor<8x!tt.ptr<i32, 1>>
    "tts.store"(%4, %3) <{static_dims = array<i64>}> : (tensor<8x!tt.ptr<i32, 1>>, tensor<8xi32>) -> ()
    tt.return
  }
}

