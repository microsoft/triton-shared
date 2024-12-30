  tt.func public @nested_use_same_level_loop_result(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
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
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
      }
      %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %29 = tt.addptr %arg9, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.addptr %29, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
        %31 = tt.addptr %30, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %32 = tt.addptr %27, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %25 = tt.addptr %24#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %25, %24#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
