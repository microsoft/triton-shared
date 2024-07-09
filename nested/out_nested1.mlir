module {
  tt.func public @nested1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
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
    %21 = arith.muli %arg5, %c32_i32 : i32
    %22 = tt.splat %21 : i32 -> tensor<2x2xi32>
    %23 = builtin.unrealized_conversion_cast %11 : tensor<2x2x!tt.ptr<f32>> to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {make_state}
    %24 = builtin.unrealized_conversion_cast %20 : tensor<2x2x!tt.ptr<f32>> to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {make_state}
    %25:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %23, %arg10 = %24) -> (tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>, tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>)  : i32 {
      %26 = builtin.unrealized_conversion_cast %arg10 : tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> to tensor<2x2x!tt.ptr<f32>> {state_placeholder}
      %27 = builtin.unrealized_conversion_cast %arg9 : tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> to tensor<2x2x!tt.ptr<f32>> {state_placeholder}
      %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
      tt.store %26, %28 : tensor<2x2x!tt.ptr<f32>>
      %29 = tt.addptr %27, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %30 = tt.addptr %26, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %31 = builtin.unrealized_conversion_cast %29 : tensor<2x2x!tt.ptr<f32>> to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {make_state}
      %32 = builtin.unrealized_conversion_cast %30 : tensor<2x2x!tt.ptr<f32>> to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {make_state}
      scf.yield %31, %32 : tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>, tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>
    }
    tt.return
  }
}

