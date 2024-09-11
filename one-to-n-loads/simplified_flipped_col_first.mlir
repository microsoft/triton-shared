module {
  tt.func public @one_to_n_loads_simplified_flipped(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32

    // making the row offsets
    // offs_x_m[:, None] * stride_x_m
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i32>>
    %10 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %11 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %12 = arith.muli %10, %11 : tensor<2x1xi32>

    // offs_x_k[None, :] * stride_x_k
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %7 = arith.muli %5, %6 : tensor<1x4xi32>

    // In + offs_x_k[None, :] * stride_x_k
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>

    // combine
    %13 = tt.broadcast %9 : tensor<1x4x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %14 = tt.broadcast %12 : tensor<2x1xi32> -> tensor<2x4xi32>
    %15 = tt.addptr %13, %14 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>


    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %17 = arith.muli %16, %11 : tensor<2x1xi32>
    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %17 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %20 = tt.broadcast %19 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %21 = tt.broadcast %7 : tensor<1x4xi32> -> tensor<2x4xi32>
    %22 = tt.addptr %20, %21 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %23 = arith.muli %arg4, %c4_i32 : i32
    %24 = tt.splat %23 : i32 -> tensor<2x4xi32>
    %25:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %15, %arg7 = %22) -> (tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>)  : i32 {
      %26 = tt.load %arg6 : tensor<2x4x!tt.ptr<f32>>
      tt.store %arg7, %26 : tensor<2x4x!tt.ptr<f32>>
      %27 = tt.addptr %arg6, %24 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %28 = tt.addptr %arg7, %24 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %27, %28 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}
