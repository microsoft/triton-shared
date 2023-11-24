module {
  tt.func public @matmul_kernel_0d1d2d34567c89c1011c(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %c15_i32 = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<16> : tensor<32x16xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c32_i32 : i32
    %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %16 = tt.splat %14 : (i32) -> tensor<32xi32>
    %17 = arith.addi %16, %15 : tensor<32xi32>
    %18 = tt.splat %arg3 : (i32) -> tensor<32xi32>
    %19 = arith.remsi %17, %18 : tensor<32xi32>
    %20 = arith.muli %13, %c64_i32 : i32
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %22 = tt.splat %20 : (i32) -> tensor<64xi32>
    %23 = arith.addi %22, %21 : tensor<64xi32>
    %24 = tt.splat %arg4 : (i32) -> tensor<64xi32>
    %25 = arith.remsi %23, %24 : tensor<64xi32>
    %26 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %27 = tt.expand_dims %19 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %28 = tt.splat %arg6 : (i32) -> tensor<32x1xi32>
    %29 = arith.muli %27, %28 : tensor<32x1xi32>
    %30 = tt.expand_dims %26 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
    %31 = tt.broadcast %29 : (tensor<32x1xi32>) -> tensor<32x16xi32>
    %32 = tt.broadcast %30 : (tensor<1x16xi32>) -> tensor<32x16xi32>
    %33 = arith.addi %31, %32 : tensor<32x16xi32>
    %34 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x16x!tt.ptr<f32, 1>>
    %35 = tt.addptr %34, %33 : tensor<32x16x!tt.ptr<f32, 1>>, tensor<32x16xi32>
    %36 = tt.expand_dims %26 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
    %37 = tt.splat %arg7 : (i32) -> tensor<16x1xi32>
    %38 = arith.muli %36, %37 : tensor<16x1xi32>
    %39 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %40 = tt.broadcast %38 : (tensor<16x1xi32>) -> tensor<16x64xi32>
    %41 = tt.broadcast %39 : (tensor<1x64xi32>) -> tensor<16x64xi32>
    %42 = arith.addi %40, %41 : tensor<16x64xi32>
    %43 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<16x64x!tt.ptr<f32, 1>>
    %44 = tt.addptr %43, %42 : tensor<16x64x!tt.ptr<f32, 1>>, tensor<16x64xi32>
    %45 = arith.addi %arg5, %c15_i32 : i32
    %46 = arith.divsi %45, %c16_i32 : i32
    %47 = arith.muli %arg7, %c16_i32 : i32
    %48 = tt.splat %47 : (i32) -> tensor<16x64xi32>
    %49:3 = scf.for %arg9 = %c0_i32 to %46 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %35, %arg12 = %44) -> (tensor<32x64xf32>, tensor<32x16x!tt.ptr<f32, 1>>, tensor<16x64x!tt.ptr<f32, 1>>)  : i32 {
      %66 = arith.muli %arg9, %c16_i32 : i32
      %67 = arith.subi %arg5, %66 : i32
      %68 = tt.splat %67 : (i32) -> tensor<1x16xi32>
      %69 = arith.cmpi slt, %30, %68 : tensor<1x16xi32>
      %70 = tt.broadcast %69 : (tensor<1x16xi1>) -> tensor<32x16xi1>
      %71 = tt.load %arg11, %70, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x16xf32>
      %72 = tt.splat %67 : (i32) -> tensor<16x1xi32>
      %73 = arith.cmpi slt, %36, %72 : tensor<16x1xi32>
      %74 = tt.broadcast %73 : (tensor<16x1xi1>) -> tensor<16x64xi1>
      %75 = tt.load %arg12, %74, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64xf32>
      %76 = tt.dot %71, %75, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
      %77 = tt.addptr %arg11, %cst : tensor<32x16x!tt.ptr<f32, 1>>, tensor<32x16xi32>
      %78 = tt.addptr %arg12, %48 : tensor<16x64x!tt.ptr<f32, 1>>, tensor<16x64xi32>
      scf.yield %76, %77, %78 : tensor<32x64xf32>, tensor<32x16x!tt.ptr<f32, 1>>, tensor<16x64x!tt.ptr<f32, 1>>
    }
    %50 = tt.expand_dims %17 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %51 = tt.splat %arg8 : (i32) -> tensor<32x1xi32>
    %52 = arith.muli %51, %50 : tensor<32x1xi32>
    %53 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>>
    %54 = tt.addptr %53, %52 : tensor<32x1x!tt.ptr<f32, 1>>, tensor<32x1xi32>
    %55 = tt.expand_dims %23 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %56 = tt.broadcast %54 : (tensor<32x1x!tt.ptr<f32, 1>>) -> tensor<32x64x!tt.ptr<f32, 1>>
    %57 = tt.broadcast %55 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %58 = tt.addptr %56, %57 : tensor<32x64x!tt.ptr<f32, 1>>, tensor<32x64xi32>
    %59 = tt.splat %arg3 : (i32) -> tensor<32x1xi32>
    %60 = arith.cmpi slt, %50, %59 : tensor<32x1xi32>
    %61 = tt.splat %arg4 : (i32) -> tensor<1x64xi32>
    %62 = arith.cmpi slt, %55, %61 : tensor<1x64xi32>
    %63 = tt.broadcast %60 : (tensor<32x1xi1>) -> tensor<32x64xi1>
    %64 = tt.broadcast %62 : (tensor<1x64xi1>) -> tensor<32x64xi1>
    %65 = arith.andi %63, %64 : tensor<32x64xi1>
    tt.store %58, %49#0, %65 {cache = 1 : i32, evict = 1 : i32} : tensor<32x64xf32>
    tt.return
  }
}
