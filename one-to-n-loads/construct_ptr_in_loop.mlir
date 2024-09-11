module {
  tt.func public @one_to_n_loads_simplified(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %4 = arith.muli %2, %3 : tensor<2x1xi32>
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %7 = arith.muli %5, %6 : tensor<1x4xi32>
    %8 = tt.broadcast %7 : tensor<1x4xi32> -> tensor<2x4xi32>
    scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %9 = tt.addptr %arg0, %arg5 : !tt.ptr<f32>, i32
      %10 = tt.splat %9 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
      %11 = tt.addptr %10, %4 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
      %12 = tt.broadcast %11 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
      %13 = tt.addptr %12, %8 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %14 = tt.addptr %arg2, %arg5 : !tt.ptr<f32>, i32
      %15 = tt.splat %14 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
      %16 = tt.addptr %15, %4 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
      %17 = tt.broadcast %16 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
      %18 = tt.addptr %17, %8 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %19 = tt.load %13 : tensor<2x4x!tt.ptr<f32>>
      tt.store %18, %19 : tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}
