module {
  tt.func public @one_to_n_loads_no_loops(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    // offs_x_m = tl.arange(0, BLOCK_M)
    // offs_x_m = tl.load(GatherIndx + offs_x_m)
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i32>>


    // dynamic dims
    // offs_x_m[:, None] * stride_x_m
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %7 = arith.muli %5, %6 : tensor<2x1xi32>

    %gather_offsets = %7 : tensor<2x1xi32>

    // static dims
    // offs_x_k[None, :] * stride_x_k
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %10 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %11 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %12 = arith.muli %10, %11 : tensor<1x4xi32>

    %static_dim_ptr = %12 : tensor<1x4xi32>

    %13 = tt.broadcast %9 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %14 = tt.broadcast %12 : tensor<1x4xi32> -> tensor<2x4xi32>

    // InPtrs = In + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
    %15 = tt.addptr %13, %14 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>

    // OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %17 = arith.muli %16, %6 : tensor<2x1xi32>
    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %17 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %20 = tt.broadcast %19 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %14 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>

    // ------
    // before
    %22 = tt.load %15 : tensor<2x4x!tt.ptr<f32>>
    // ------
    // after
    %22 = tts.structured_gather %gather_offsets %static_dim_ptr {axis = 0} tensor<2x4x!tt.ptr<f32>>

    tt.store %21, %22 : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}
