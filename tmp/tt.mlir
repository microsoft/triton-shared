module {
  tt.func public @wrap_side_by_side_masked_loop_0d1d2e3e4e5c6e7c(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<4> : tensor<4x4xi32>
    %c4_i32 = arith.constant 4 : i32
    %cst_1 = arith.constant dense<2> : tensor<4x1xi32>
    %cst_2 = arith.constant dense<6> : tensor<4xi32>
    %cst_3 = arith.constant dense<2> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_3 : tensor<4xi32>
    %2 = arith.addi %0, %cst_2 : tensor<4xi32>
    %3 = tt.splat %arg3 : (i32) -> tensor<4xi32>
    %4 = arith.remsi %2, %3 : tensor<4xi32>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : (i32) -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
    %9 = tt.broadcast %7 : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %10 = tt.broadcast %8 : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %11 = arith.addi %9, %10 : tensor<4x4xi32>
    %12 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<4x4x!tt.ptr<f32, 1>>
    %13 = tt.addptr %12, %11 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
    %14 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %15 = tt.splat %arg5 : (i32) -> tensor<4x1xi32>
    %16 = arith.muli %15, %14 : tensor<4x1xi32>
    %17 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<4x1x!tt.ptr<f32, 1>>
    %18 = tt.addptr %17, %16 : tensor<4x1x!tt.ptr<f32, 1>>, tensor<4x1xi32>
    %19 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
    %20 = tt.broadcast %18 : (tensor<4x1x!tt.ptr<f32, 1>>) -> tensor<4x4x!tt.ptr<f32, 1>>
    %21 = tt.broadcast %19 : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %22 = tt.addptr %20, %21 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
    %23 = arith.cmpi slt, %14, %cst_1 : tensor<4x1xi32>
    %24 = tt.broadcast %23 : (tensor<4x1xi1>) -> tensor<4x4xi1>
    %25 = arith.muli %arg4, %c4_i32 : i32
    %26 = tt.splat %25 : (i32) -> tensor<4x4xi32>
    %27:2 = scf.for %arg6 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg7 = %13, %arg8 = %22) -> (tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4x!tt.ptr<f32, 1>>)  : i32 {
      %28 = tt.load %arg7, %24, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x4xf32>
      tt.store %arg8, %28 {cache = 1 : i32, evict = 1 : i32} : tensor<4x4xf32>
      %29 = tt.addptr %arg7, %26 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
      %30 = tt.addptr %arg8, %cst_0 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
      scf.yield %29, %30 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4x!tt.ptr<f32, 1>>
    }
    tt.return
  }
}
