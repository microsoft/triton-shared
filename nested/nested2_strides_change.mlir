module {
  tt.func public @nested2_strides_change(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
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
    %16 = arith.addi %arg3, %c1_i32 : i32
    %17 = arith.muli %16, %c4_i32 : i32
    %18 = tt.splat %17 : i32 -> tensor<2x2xi32>
    %19 = arith.muli %arg3, %c4_i32 : i32
    %20 = tt.splat %19 : i32 -> tensor<2x2xi32>
    %21 = arith.addi %arg3, %c2_i32 : i32
    %22 = arith.muli %21, %c4_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x2xi32>
    %24:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %25 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg6, %25 : tensor<2x2x!tt.ptr<f32>>
      %26 = tt.addptr %arg5, %18 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %27 = tt.addptr %arg6, %18 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %28:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %26, %arg9 = %27) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %31 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %31 : tensor<2x2x!tt.ptr<f32>>
        %32 = tt.addptr %arg8, %20 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %33 = tt.addptr %arg9, %20 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %33 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %29 = tt.addptr %28#0, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %30 = tt.addptr %28#1, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %29, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}
