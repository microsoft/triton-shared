#loc = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":342:0)
module {
  tt.func public @triton__0d1d2de(%arg0: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32} loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":342:0), %arg1: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32} loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":342:0), %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32} loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":342:0)) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1x64xi32> loc(#loc1)
    %cst_0 = arith.constant dense<77> : tensor<4x64xi32> loc(#loc1)
    %cst_1 = arith.constant dense<128> : tensor<4x1xi32> loc(#loc1)
    %cst_2 = arith.constant dense<32> : tensor<1x64xi32> loc(#loc1)
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c4_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> loc(#loc4)
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32> loc(#loc5)
    %4 = tt.splat %1 : (i32) -> tensor<4x1xi32> loc(#loc6)
    %5 = arith.addi %4, %3 : tensor<4x1xi32> loc(#loc6)
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> loc(#loc7)
    %7 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32> loc(#loc8)
    %8 = arith.cmpi slt, %7, %cst_2 : tensor<1x64xi32> loc(#loc9)
    %9 = arith.muli %5, %cst_1 : tensor<4x1xi32> loc(#loc10)
    %10 = tt.broadcast %7 : (tensor<1x64xi32>) -> tensor<4x64xi32> loc(#loc11)
    %11 = tt.broadcast %9 : (tensor<4x1xi32>) -> tensor<4x64xi32> loc(#loc11)
    %12 = arith.addi %10, %11 : tensor<4x64xi32> loc(#loc11)
    %13 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<4x64x!tt.ptr<i32, 1>> loc(#loc12)
    %14 = tt.addptr %13, %12 : tensor<4x64x!tt.ptr<i32, 1>>, tensor<4x64xi32> loc(#loc12)
    %15 = tt.broadcast %8 : (tensor<1x64xi1>) -> tensor<4x64xi1> loc(#loc13)
    %16 = tt.load %14, %15, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x64xi32> loc(#loc13)
    %17 = arith.muli %7, %cst : tensor<1x64xi32> loc(#loc14)
    %18 = tt.broadcast %17 : (tensor<1x64xi32>) -> tensor<4x64xi32> loc(#loc15)
    %19 = tt.broadcast %3 : (tensor<4x1xi32>) -> tensor<4x64xi32> loc(#loc15)
    %20 = arith.addi %18, %19 : tensor<4x64xi32> loc(#loc15)
    %21 = tt.splat %arg1 : (!tt.ptr<i32, 1>) -> tensor<4x64x!tt.ptr<i32, 1>> loc(#loc16)
    %22 = tt.addptr %21, %20 : tensor<4x64x!tt.ptr<i32, 1>>, tensor<4x64xi32> loc(#loc16)
    tt.store %22, %16 {cache = 1 : i32, evict = 1 : i32} : tensor<4x64xi32> loc(#loc17)
    tt.return loc(#loc18)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":346:28)
#loc3 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":346:33)
#loc4 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":347:36)
#loc5 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":347:44)
#loc6 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":347:23)
#loc7 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":348:25)
#loc8 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":348:33)
#loc9 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":353:21)
#loc10 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":355:45)
#loc11 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":355:36)
#loc12 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":355:30)
#loc13 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":355:51)
#loc14 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":357:29)
#loc15 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":358:20)
#loc16 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":357:20)
#loc17 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":358:52)
#loc18 = loc("/home/nhat/github/triton_shared/python/examples/test_modulo.py":356:4)