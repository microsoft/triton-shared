module {
  tt.func @kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3:2 = scf.for %arg5 = %c0 to %c12 step %c3 iter_args(%arg6 = %cst, %arg7 = %2) -> (tensor<128x128xf32>, index) {
      %8 = arith.addi %arg7, %c128 : index
      %9 = tts.make_tptr %arg1 to sizes: [128, 128], strides: [1, 1], offsets: [%8, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
      %10 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
      %11 = math.exp %10 : tensor<128x128xf32>
      %12 = arith.addf %arg6, %11 : tensor<128x128xf32>
      %13 = arith.addi %arg7, %arg5 : index
      scf.yield %12, %13 : tensor<128x128xf32>, index
    }
    %4 = arith.muli %0, %arg3 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.addi %5, %c128 : index
    %7 = tts.make_tptr %arg0 to sizes: [128, 128], strides: [1, 1], offsets: [%6, 0], shape: [0, 0], order: [] : <f32, 1> to tensor<128x128x!tt.ptr<f32, 1>>
    "tts.store"(%7, %3#0) <{static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>, tensor<128x128xf32>) -> ()
    tt.return
  }
}

