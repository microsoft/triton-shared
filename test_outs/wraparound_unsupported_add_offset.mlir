module {
  tt.func public @wrap_side_by_side_masked_loop_01234567(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<6> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg3 : (i32) -> tensor<4xi32>
    %3 = arith.remsi %0, %2 : tensor<4xi32>
    %4 = arith.addi %3, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : (i32) -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
    %9 = tt.splat %arg5 : (i32) -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<4x4x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %17 = arith.index_cast %arg6 : i32 to index
    %18 = arith.index_cast %arg7 : i32 to index
    %19 = arith.cmpi slt, %16, %cst_0 : tensor<4x1xi32>
    %20 = tt.broadcast %19 : (tensor<4x1xi1>) -> tensor<4x4xi1>
    %21 = arith.muli %arg4, %c4_i32 : i32
    %22 = tt.splat %21 : (i32) -> tensor<4x4xi32>
    %23 = arith.muli %arg5, %c4_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %c0) -> (tensor<4x4x!tt.ptr<f32, 1>>, index)  : i32 {
      %26 = tts.make_tptr %arg1 to sizes: [4, 4], strides: [%17, %18], offsets: [%arg10, %c0], shape: [0, 0], order: [] : <f32, 1> to tensor<4x4x!tt.ptr<f32, 1>>
      %27 = tt.load %arg9, %20, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x4xf32>
      "tts.store"(%26, %27) <{static_dims = array<i64>}> : (tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xf32>) -> ()
      %28 = tt.addptr %arg9, %22 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
      %29 = arith.addi %arg10, %24 : index
      scf.yield %28, %29 : tensor<4x4x!tt.ptr<f32, 1>>, index
    }
    tt.return
  }
}

