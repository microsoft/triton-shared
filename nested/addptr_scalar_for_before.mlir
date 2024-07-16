module {
  tt.func @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3:2 = scf.for %arg5 = %c0 to %c12 step %c3 iter_args(%arg6 = %cst, %arg7 = %2) -> (tensor<1024xf32>, index) {
      %7 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%arg7], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
      %8 = "tts.load"(%7) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
      %9 = math.exp %8 : tensor<1024xf32>
      %10 = arith.addf %arg6, %9 : tensor<1024xf32>
      %11 = arith.addi %arg7, %arg5 : index
      scf.yield %10, %11 : tensor<1024xf32>, index
    }
    %4 = arith.muli %0, %arg3 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%5], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%6, %3#0) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    tt.return
  }
}
