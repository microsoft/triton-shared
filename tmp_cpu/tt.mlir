module {
  tt.func public @triton__0d1d2de(%arg0: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1x64xi32>
    %cst_0 = arith.constant dense<77> : tensor<4x64xi32>
    %cst_1 = arith.constant dense<128> : tensor<4x1xi32>
    %cst_2 = arith.constant dense<32> : tensor<1x64xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    %4 = tt.splat %1 : (i32) -> tensor<4x1xi32>
    %5 = arith.addi %4, %3 : tensor<4x1xi32>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %8 = arith.cmpi slt, %7, %cst_2 : tensor<1x64xi32>
    %9 = arith.muli %5, %cst_1 : tensor<4x1xi32>
    %10 = tt.broadcast %7 : (tensor<1x64xi32>) -> tensor<4x64xi32>
    %11 = tt.broadcast %9 : (tensor<4x1xi32>) -> tensor<4x64xi32>
    %12 = arith.addi %10, %11 : tensor<4x64xi32>
    %13 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<4x64x!tt.ptr<i32, 1>>
    %14 = tt.addptr %13, %12 : tensor<4x64x!tt.ptr<i32, 1>>, tensor<4x64xi32>
    %15 = tt.broadcast %8 : (tensor<1x64xi1>) -> tensor<4x64xi1>
    %16 = tt.load %14, %15, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x64xi32>
    %17 = arith.muli %7, %cst : tensor<1x64xi32>
    %18 = tt.broadcast %17 : (tensor<1x64xi32>) -> tensor<4x64xi32>
    %19 = tt.broadcast %3 : (tensor<4x1xi32>) -> tensor<4x64xi32>
    %20 = arith.addi %18, %19 : tensor<4x64xi32>
    %21 = tt.splat %arg1 : (!tt.ptr<i32, 1>) -> tensor<4x64x!tt.ptr<i32, 1>>
    %22 = tt.addptr %21, %20 : tensor<4x64x!tt.ptr<i32, 1>>, tensor<4x64xi32>
    tt.store %22, %16 {cache = 1 : i32, evict = 1 : i32} : tensor<4x64xi32>
    tt.return
  }
}
