module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c1024 = arith.constant 1024 : index
    %0:5 = scf.for %arg2 = %c0 to %c12 step %c3 iter_args(%arg3 = %c1, %arg4 = %c2, %arg5 = %c3, %arg6 = %c1024, %arg7 = %c1024) -> (index, index, index, index, index) {
      %1 = tts.make_tptr %arg1 to sizes: [256], strides: [%c1], offsets: [%arg7], shape: [0], order: [] : <bf16, 1> to tensor<256x!tt.ptr<bf16, 1>>
      %2 = tts.make_tptr %arg0 to sizes: [256], strides: [%c1], offsets: [%arg6], shape: [0], order: [] : <bf16, 1> to tensor<256x!tt.ptr<bf16, 1>>
      %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16, 1>>) -> tensor<256xbf16>
      "tts.store"(%1, %3) <{static_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16, 1>>, tensor<256xbf16>) -> ()
      %4 = arith.addi %arg6, %c3 : index
      %5 = arith.addi %arg3, %c3 : index
      %6 = arith.addi %arg4, %c3 : index
      %7 = arith.addi %arg5, %c3 : index
      %8 = arith.addi %5, %6 : index
      %9 = arith.addi %8, %7 : index
      %10 = arith.addi %arg7, %9 : index
      scf.yield %5, %6, %7, %4, %10 : index, index, index, index, index
    }
    tt.return
  }
}

