module {
  tt.func public @nested_strides_change(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
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
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg6 : i32 -> tensor<2x1xi32>
    %13 = arith.muli %12, %1 : tensor<2x1xi32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %16 = tt.splat %arg7 : i32 -> tensor<1x2xi32>
    %17 = arith.muli %16, %4 : tensor<1x2xi32>
    %18 = tt.broadcast %15 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %19 = tt.broadcast %17 : tensor<1x2xi32> -> tensor<2x2xi32>
    %20 = tt.addptr %18, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %21 = arith.addi %arg5, %c1_i32 : i32
    %22 = arith.muli %21, %c32_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x2xi32>
    %24 = arith.muli %arg5, %c32_i32 : i32
    %25 = tt.splat %24 : i32 -> tensor<2x2xi32>
    %26:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %11, %arg10 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %27 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %27 : tensor<2x2x!tt.ptr<f32>>
      %28 = tt.addptr %arg9, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %29 = tt.addptr %arg10, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %30:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %28, %arg13 = %29) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %31 = tt.load %arg12 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg13, %31 : tensor<2x2x!tt.ptr<f32>>
        %32 = tt.addptr %arg12, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %33 = tt.addptr %arg13, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %33 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      scf.yield %30#0, %30#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}