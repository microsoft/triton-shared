module {
  tt.func public @wrap_side_by_side_loop_0d1d23e4e5c6e7c(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<6> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst : tensor<4xi32>
    %2 = tt.splat %arg3 : (i32) -> tensor<4xi32>
    %3 = arith.remsi %1, %2 : tensor<4xi32>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %5 = tt.splat %arg4 : (i32) -> tensor<4x1xi32>
    %6 = arith.muli %4, %5 : tensor<4x1xi32>
    %7 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
    %8 = tt.broadcast %6 : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %9 = tt.broadcast %7 : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %10 = arith.addi %8, %9 : tensor<4x4xi32>
    %11 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<4x4x!tt.ptr<f32, 1>>
    %12 = tt.addptr %11, %10 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
    %13 = tt.splat %arg5 : (i32) -> tensor<4x1xi32>
    %14 = arith.muli %13, %4 : tensor<4x1xi32>
    %15 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<4x1x!tt.ptr<f32, 1>>
    %16 = tt.addptr %15, %14 : tensor<4x1x!tt.ptr<f32, 1>>, tensor<4x1xi32>
    %17 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
    %18 = tt.broadcast %16 : (tensor<4x1x!tt.ptr<f32, 1>>) -> tensor<4x4x!tt.ptr<f32, 1>>
    %19 = tt.broadcast %17 : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %20 = tt.addptr %18, %19 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
    %21 = arith.muli %arg4, %c4_i32 : i32
    %22 = tt.splat %21 : (i32) -> tensor<4x4xi32>
    %23:2 = scf.for %arg6 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg7 = %12, %arg8 = %20) -> (tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4x!tt.ptr<f32, 1>>)  : i32 {
      %24 = tt.load %arg7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x4xf32>
      tt.store %arg8, %24 {cache = 1 : i32, evict = 1 : i32} : tensor<4x4xf32>
      %25 = tt.addptr %arg7, %22 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
      %26 = tt.addptr %arg8, %22 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4xi32>
      scf.yield %25, %26 : tensor<4x4x!tt.ptr<f32, 1>>, tensor<4x4x!tt.ptr<f32, 1>>
    }
    tt.return
  }
}
