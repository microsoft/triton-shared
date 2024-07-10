build cast op2
result type
tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>
inputs:
<block argument> of type 'tensor<2x2x!tt.ptr<f32>>' at index: 6
<block argument> of type 'index' at index: 7
<block argument> of type 'index' at index: 8
<block argument> of type 'index' at index: 9
<block argument> of type 'index' at index: 10
build cast op2
result type
tuple<tensor<2x2x!tt.ptr<f32>>, tuple<index, index, index, index>>
inputs:
<block argument> of type 'tensor<2x2x!tt.ptr<f32>>' at index: 1
<block argument> of type 'index' at index: 2
<block argument> of type 'index' at index: 3
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
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
      %26 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg14, %26 : tensor<2x2x!tt.ptr<f32>>
      %27 = tt.addptr %arg9, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %28 = tt.addptr %arg14, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %29:5 = builtin.unrealized_conversion_cast %27 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
      %30:5 = builtin.unrealized_conversion_cast %28 : tensor<2x2x!tt.ptr<f32>> to tensor<2x2x!tt.ptr<f32>>, index, index, index, index {make_state_new}
      scf.yield %29#0, %29#1, %29#2, %29#3, %29#4, %30#0, %30#1, %30#2, %30#3, %30#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
    }
    tt.return
  }
}

