module {
  tt.func public @nested2_use_loop_results(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
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
    %16 = arith.muli %arg3, %c4_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:5 = "tts.state_placeholder"(%11) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
    %19:5 = "tts.state_placeholder"(%15) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
    %20:10 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %18#0, %arg6 = %18#1, %arg7 = %18#2, %arg8 = %18#3, %arg9 = %18#4, %arg10 = %19#0, %arg11 = %19#1, %arg12 = %19#2, %arg13 = %19#3, %arg14 = %19#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
      %21 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %21 : tensor<2x2x!tt.ptr<f32>>
      %22 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %23 = tt.addptr %arg10, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24:5 = "tts.state_placeholder"(%22) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
      %25:5 = "tts.state_placeholder"(%23) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
      %26:10 = scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg16 = %24#0, %arg17 = %24#1, %arg18 = %24#2, %arg19 = %24#3, %arg20 = %24#4, %arg21 = %25#0, %arg22 = %25#1, %arg23 = %25#2, %arg24 = %25#3, %arg25 = %25#4) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index)  : i32 {
        %31 = tt.load %arg16 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg21, %31 : tensor<2x2x!tt.ptr<f32>>
        %32 = tt.addptr %arg16, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %33 = tt.addptr %arg21, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %34:5 = "tts.state_placeholder"(%32) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
        %35:5 = "tts.state_placeholder"(%33) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
        scf.yield %34#0, %34#1, %34#2, %34#3, %34#4, %35#0, %35#1, %35#2, %35#3, %35#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
      }
      %27 = tt.addptr %26#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %28 = tt.addptr %26#5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %29:5 = "tts.state_placeholder"(%27) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
      %30:5 = "tts.state_placeholder"(%28) : (tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2x!tt.ptr<f32>>, index, index, index, index)
      scf.yield %29#0, %29#1, %29#2, %29#3, %29#4, %30#0, %30#1, %30#2, %30#3, %30#4 : tensor<2x2x!tt.ptr<f32>>, index, index, index, index, tensor<2x2x!tt.ptr<f32>>, index, index, index, index
    }
    tt.return
  }
}
