/home/nhat/github/triton_shared/nested/nested1.mlir:1:1: error: failed to apply conversion patterns
module {
^
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
    %23:5 = builtin.unrealized_conversion_cast %11 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
    %24:5 = builtin.unrealized_conversion_cast %20 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
    %25:10 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %23#0, %arg10 = %23#1, %arg11 = %23#2, %arg12 = %23#3, %arg13 = %23#4, %arg14 = %24#0, %arg15 = %24#1, %arg16 = %24#2, %arg17 = %24#3, %arg18 = %24#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
      %26 = builtin.unrealized_conversion_cast %arg14, %arg15, %arg16, %arg17, %arg18 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {"__one-to-n-type-conversion_cast-kind__" = "argument"}
      %27 = builtin.unrealized_conversion_cast %arg9, %arg10, %arg11, %arg12, %arg13 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index to tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> {"__one-to-n-type-conversion_cast-kind__" = "argument"}
      %28 = builtin.unrealized_conversion_cast %26 : tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> to tensor<2x2x!tt.ptr<f32>> {state_placeholder}
      %29 = builtin.unrealized_conversion_cast %27 : tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>> to tensor<2x2x!tt.ptr<f32>> {state_placeholder}
      %30 = tt.load %29 : tensor<2x2x!tt.ptr<f32>>
      tt.store %28, %30 : tensor<2x2x!tt.ptr<f32>>
      %31 = tt.addptr %29, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %32 = tt.addptr %28, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %33:5 = builtin.unrealized_conversion_cast %31 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
      %34:5 = builtin.unrealized_conversion_cast %32 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
      scf.yield %33#0, %33#1, %33#2, %33#3, %33#4, %34#0, %34#1, %34#2, %34#3, %34#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
    }
    tt.return
  }
}

