#loc = loc("<ipython-input-6-d73b1d3bb267>":94:0)
module {
  tt.func public @mask(%arg0: !tt.ptr<f32> loc("<ipython-input-6-d73b1d3bb267>":94:0), %arg1: !tt.ptr<f32> loc("<ipython-input-6-d73b1d3bb267>":94:0)) attributes {noinline = false} {
    %c10_i32 = arith.constant 10 : i32 loc(#loc1)
    %cst = arith.constant dense<12> : tensor<16x16xi32> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> loc(#loc4)
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> loc(#loc5)
    %4 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32> loc(#loc6)
    %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32> loc(#loc6)
    %6 = arith.addi %4, %5 : tensor<16x16xi32> loc(#loc6)
    %7 = arith.cmpi slt, %0, %c10_i32 : i32 loc(#loc7)
    %8 = arith.cmpi slt, %6, %cst : tensor<16x16xi32> loc(#loc8)
    %9 = tt.splat %7 : i1 -> tensor<16x16xi1> loc(#loc9)
    %10 = arith.andi %9, %8 : tensor<16x16xi1> loc(#loc9)
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>> loc(#loc10)
    %12 = tt.addptr %11, %6 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32> loc(#loc10)
    %13 = tt.load %12, %10 : tensor<16x16x!tt.ptr<f32>> loc(#loc11)
    // tt.print " hi: " {hex = false} : %13 : tensor<16x16xf32> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("<ipython-input-6-d73b1d3bb267>":95:20)
#loc3 = loc("<ipython-input-6-d73b1d3bb267>":97:22)
#loc4 = loc("<ipython-input-6-d73b1d3bb267>":97:26)
#loc5 = loc("<ipython-input-6-d73b1d3bb267>":97:54)
#loc6 = loc("<ipython-input-6-d73b1d3bb267>":97:37)
#loc7 = loc("<ipython-input-6-d73b1d3bb267>":98:23)
#loc8 = loc("<ipython-input-6-d73b1d3bb267>":98:37)
#loc9 = loc("<ipython-input-6-d73b1d3bb267>":98:30)
#loc10 = loc("<ipython-input-6-d73b1d3bb267>":99:19)
#loc11 = loc("<ipython-input-6-d73b1d3bb267>":99:14)
#loc12 = loc("<ipython-input-6-d73b1d3bb267>":100:24)
#loc13 = loc("<ipython-input-6-d73b1d3bb267>":100:2)