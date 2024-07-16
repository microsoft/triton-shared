module {
  tt.func @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %4:3 = scf.for %arg5 = %c0 to %c12 step %c3 iter_args(%arg6 = %3, %arg7 = %2, %arg8 = %cst) -> (!tt.ptr<f32>, index, tensor<1024xf32>) {
      %12 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %13 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %14 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%arg7], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
      %15 = tt.addptr %13, %12 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %16 = "tts.load"(%14) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
      %17 = math.exp %16 : tensor<1024xf32>
      %18 = arith.addf %arg8, %17 : tensor<1024xf32>
      %19 = arith.index_cast %arg5 : index to i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.addi %arg7, %20 : index
      %22 = tt.addptr %arg6, %19 : !tt.ptr<f32>, i32
      scf.yield %22, %21, %18 : !tt.ptr<f32>, index, tensor<1024xf32>
    }
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %6 = arith.muli %0, %arg3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = tt.addptr %arg0, %6 : !tt.ptr<f32>, i32
    %9 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%7], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %9, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    "tts.store"(%10, %4#2) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    tt.return
  }
}

