module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>, %arg3: i32, %arg4: i32) {
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, %c5], offsets: [%0, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    %2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
    %3 = arith.index_cast %arg3 : i32 to index
    %4:2 = scf.for %arg5 = %c0 to %c12 step %c3 iter_args(%arg6 = %2, %arg7 = %3) -> (tensor<4x256xbf16>, index) {
      %7 = tts.make_tptr %arg1 to sizes: [4, 256], strides: [%c1, %c5], offsets: [%arg7, %c0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
      %8 = "tts.load"(%7) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
      %9 = arith.addf %arg6, %8 : tensor<4x256xbf16>
      %10 = arith.addi %arg7, %c3 : index
      scf.yield %9, %10 : tensor<4x256xbf16>, index
    }
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = tts.make_tptr %arg2 to sizes: [4, 256], strides: [1, %c5], offsets: [%5, 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
    "tts.store"(%6, %4#0) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

