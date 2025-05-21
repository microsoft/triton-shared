// RUN: triton-shared-opt --triton-to-structured="run-prepass-only=true" --split-input-file %s | FileCheck %s

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
    %23:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %11, %arg10 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %24 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %24 : tensor<2x2x!tt.ptr<f32>>
      %25 = tt.addptr %arg9, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %26 = tt.addptr %arg10, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %25, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-COUNT-4: tts.get_structured_state
// CHECK-NOT: tts.get_structured_state

// ----

module {
  tt.func public @nested2(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
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
    %23:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %11, %arg10 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %24 = tt.load %arg9 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg10, %24 : tensor<2x2x!tt.ptr<f32>>
      %25 = tt.addptr %arg9, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %26 = tt.addptr %arg10, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %27:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %25, %arg13 = %26) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %28 = tt.load %arg12 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg13, %28 : tensor<2x2x!tt.ptr<f32>>
        %29 = tt.addptr %arg12, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.addptr %arg13, %22 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %29, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      scf.yield %27#0, %27#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-COUNT-6: tts.get_structured_state
// CHECK-NOT: tts.get_structured_state

// ----

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
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg6, %19 : tensor<2x2x!tt.ptr<f32>>
      %20 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %20, %arg9 = %21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %25 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %25 : tensor<2x2x!tt.ptr<f32>>
        %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %27 = tt.addptr %arg9, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %26, %27 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %23 = tt.addptr %22#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %22#1, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %24 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-COUNT-8: tts.get_structured_state
// CHECK-NOT: tts.get_structured_state
// ----

module {
  tt.func public @nested3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
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
    %20:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %21 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %24 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %25 = tt.load %24 : tensor<2x2x!tt.ptr<f32>>
        %26:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %24, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %27 = tt.addptr %arg11, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg12, %21 : tensor<2x2x!tt.ptr<f32>>
          %29 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %29, %25 : tensor<2x2x!tt.ptr<f32>>
          %30 = tt.addptr %29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
          %31 = tt.addptr %30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          scf.yield %27, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        scf.yield %26#0, %26#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %23 = tt.addptr %22#0, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %22#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-COUNT-6: tts.get_structured_state
// CHECK-NOT: tts.get_structured_state
