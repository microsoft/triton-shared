module {
  tt.func public @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<3> : tensor<2x2xi32>
    %cst_0 = arith.constant dense<1> : tensor<2x2xi32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg2, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.addptr %arg5, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %20 = tt.addptr %arg6, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %19, %arg9 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.addptr %arg9, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %27, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %22 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %23 = tt.addptr %22, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %25 = tt.addptr %24, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}
