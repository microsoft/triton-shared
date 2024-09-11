module {
  tt.func public @one_to_n_loads_simplified_flipped(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %3 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %4 = arith.muli %2, %3 : tensor<1x4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
    %7 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %8 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %9 = arith.muli %7, %8 : tensor<2x1xi32>
    %10 = tt.broadcast %6 : tensor<1x4x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %11 = tt.broadcast %9 : tensor<2x1xi32> -> tensor<2x4xi32>
    %12 = tt.addptr %10, %11 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %14 = tt.addptr %13, %9 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %15 = tt.broadcast %14 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %16 = tt.broadcast %4 : tensor<1x4xi32> -> tensor<2x4xi32>
    %17 = tt.addptr %15, %16 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %18 = arith.muli %arg4, %c4_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x4xi32>
    %20:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %12, %arg7 = %17) -> (tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>)  : i32 {
      %21 = tt.load %arg6 : tensor<2x4x!tt.ptr<f32>>
      tt.store %arg7, %21 : tensor<2x4x!tt.ptr<f32>>
      %22 = tt.addptr %arg6, %19 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %23 = tt.addptr %arg7, %19 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %22, %23 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}