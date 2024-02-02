module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>) {
    %c6144 = arith.constant 6144 : index
    %cst = arith.constant dense<6> : tensor<256x128xi32>
    %c6 = arith.constant 6 : index
    %0 = tt.make_range {end = 768 : i32, start = 512 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
    %2 = tt.broadcast %1 : (tensor<256x1xi32>) -> tensor<256x128xi32>
    %3 = tt.make_range {end = 1152 : i32, start = 1024 : i32} : tensor<128xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
    %5 = tt.broadcast %4 : (tensor<1x128xi32>) -> tensor<256x128xi32>
    %6 = arith.muli %5, %cst : tensor<256x128xi32>
    %7 = arith.addi %2, %6 : tensor<256x128xi32>
    %8 = tts.make_tptr %arg1 to sizes: [256, 128], strides: [1, %c6], offsets: [512, %c6144], shape: [0, 0], order: [] : <bf16, 1> to tensor<256x128x!tt.ptr<bf16, 1>>
    %9 = "tts.load"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>) -> tensor<256x128xbf16>
    "tts.store"(%8, %9) <{static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>, tensor<256x128xbf16>) -> ()
    %10 = arith.sitofp %7 : tensor<256x128xi32> to tensor<256x128xbf16>
    "tts.store"(%8, %10) <{static_dims = array<i64>}> : (tensor<256x128x!tt.ptr<bf16, 1>>, tensor<256x128xbf16>) -> ()
    tt.return
  }
}

