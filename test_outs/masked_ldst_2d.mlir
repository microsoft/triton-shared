module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32, %arg3: i32) {
    %c3072 = arith.constant 3072 : index
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c259 = arith.constant 259 : index
    %c130 = arith.constant 130 : index
    %cst = arith.constant 0xFF80 : bf16
    %0 = tts.make_tptr %arg0 to sizes: [128, 256], strides: [1, %c1024], offsets: [%c2, %c3072], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %1 = tts.make_tptr %arg1 to sizes: [128, 256], strides: [1, %c1024], offsets: [%c2, %c3072], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.minsi %2, %c130 : index
    %4 = arith.subi %3, %c2 : index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.minsi %5, %c259 : index
    %7 = arith.subi %6, %c3 : index
    %8 = arith.minsi %4, %c128 : index
    %9 = arith.minsi %7, %c256 : index
    %10 = "tts.load"(%0, %8, %9, %cst) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, index, index, bf16) -> tensor<128x256xbf16>
    %11 = arith.index_cast %arg2 : i32 to index
    %12 = arith.minsi %11, %c130 : index
    %13 = arith.subi %12, %c2 : index
    %14 = arith.index_cast %arg3 : i32 to index
    %15 = arith.minsi %14, %c259 : index
    %16 = arith.subi %15, %c3 : index
    %17 = arith.minsi %13, %c128 : index
    %18 = arith.minsi %16, %c256 : index
    "tts.store"(%1, %10, %17, %18) <{static_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>, index, index) -> ()
    tt.return
  }
}

