module {
  tt.func public @one_to_n_loads_simplified(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i32>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %6 = arith.muli %4, %5 : tensor<2x1xi32>
    %7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = arith.index_cast %arg4 : i32 to index
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<2x4xi32>
    %13 = "tts.make_gather_tptr"(%arg0, %6, %11) <{sizes = array<i64: 1, 4>, static_strides = array<i64: 0, -9223372036854775808>}> : (!tt.ptr<f32>, tensor<2x1xi32>, index) -> tensor<2x4x!tt.ptr<f32>>
    %14 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %15 = arith.muli %14, %5 : tensor<2x1xi32>
    %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %17 = tt.addptr %16, %15 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %18 = tt.broadcast %17 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %19 = tt.addptr %18, %12 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %20 = arith.muli %arg4, %c4_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x4xi32>
    %22:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %13, %arg7 = %19) -> (tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>)  : i32 {
      %23 = tt.load %arg6 : tensor<2x4x!tt.ptr<f32>>
      tt.store %arg7, %23 : tensor<2x4x!tt.ptr<f32>>
      %24 = tt.addptr %arg6, %21 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %25 = tt.addptr %arg7, %21 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %24, %25 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}