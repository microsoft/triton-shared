module {
  tt.func public @nested1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg4 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg5 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = arith.index_cast %arg4 : i32 to index
    %12 = arith.index_cast %arg5 : i32 to index
    %13 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %14 = tt.splat %arg6 : i32 -> tensor<2x1xi32>
    %15 = arith.muli %14, %1 : tensor<2x1xi32>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %17 = arith.index_cast %arg6 : i32 to index
    %18 = tt.addptr %16, %15 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %19 = tt.splat %arg7 : i32 -> tensor<1x2xi32>
    %20 = arith.muli %19, %4 : tensor<1x2xi32>
    %21 = tt.broadcast %18 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %22 = tt.broadcast %20 : tensor<1x2xi32> -> tensor<2x2xi32>
    %23 = arith.index_cast %arg7 : i32 to index
    %24 = tt.addptr %21, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %25 = arith.muli %arg5, %c32_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.index_cast %25 : i32 to index
    %28 = tt.splat %25 : i32 -> tensor<2x2xi32>
    %29:4 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %13, %arg10 = %c0, %arg11 = %24, %arg12 = %c0) -> (tensor<2x2x!tt.ptr<f32>>, index, tensor<2x2x!tt.ptr<f32>>, index)  : i32 {
      %30 = tts.make_tptr %arg1 to sizes: [2, 2], strides: [%17, %23], offsets: [%arg12, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %31 = tts.make_tptr %arg0 to sizes: [2, 2], strides: [%11, %12], offsets: [%arg10, %c0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
      %32 = "tts.load"(%31) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
      "tts.store"(%30, %32) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
      %33 = arith.addi %arg10, %27 : index
      %34 = tt.addptr %arg9, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %35 = arith.addi %arg12, %26 : index
      %36 = tt.addptr %arg11, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %34, %33, %36, %35 : tensor<2x2x!tt.ptr<f32>>, index, tensor<2x2x!tt.ptr<f32>>, index
    }
    tt.return
  }
}

