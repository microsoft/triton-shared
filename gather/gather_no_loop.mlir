#loc = loc("<ipython-input-3-51dd6ad3168f>":85:0)
module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32> loc("<ipython-input-3-51dd6ad3168f>":85:0), %arg1: !tt.ptr<f32> loc("<ipython-input-3-51dd6ad3168f>":85:0)) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<64xi32> loc(#loc1)
    %cst_0 = arith.constant dense<10> : tensor<64xi32> loc(#loc1)
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> loc(#loc2)
    %1 = arith.divsi %0, %cst_0 : tensor<64xi32> loc(#loc3)
    %2 = arith.addi %1, %cst : tensor<64xi32> loc(#loc4)
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>> loc(#loc5)
    %4 = tt.addptr %3, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32> loc(#loc5)
    %5 = tt.load %4 : tensor<64x!tt.ptr<f32>> loc(#loc6)
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>> loc(#loc7)
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32> loc(#loc7)
    tt.store %7, %5 : tensor<64x!tt.ptr<f32>> loc(#loc8)
    tt.return loc(#loc9)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("<ipython-input-3-51dd6ad3168f>":86:24)
#loc3 = loc("<ipython-input-3-51dd6ad3168f>":88:19)
#loc4 = loc("<ipython-input-3-51dd6ad3168f>":88:24)
#loc5 = loc("<ipython-input-3-51dd6ad3168f>":89:22)
#loc6 = loc("<ipython-input-3-51dd6ad3168f>":89:16)
#loc7 = loc("<ipython-input-3-51dd6ad3168f>":90:20)
#loc8 = loc("<ipython-input-3-51dd6ad3168f>":90:30)
#loc9 = loc("<ipython-input-3-51dd6ad3168f>":92:4)
