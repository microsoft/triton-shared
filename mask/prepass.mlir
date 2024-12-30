module {
  tt.func public @triton_(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant dense<9.99999974E-6> : tensor<1x1xf32>
    %cst_0 = arith.constant dense<1.000000e+01> : tensor<1x1xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x16xf32>
    %cst_2 = arith.constant dense<10> : tensor<1x16xi32>
    %cst_3 = arith.constant dense<10> : tensor<1x1xi32>
    %0 = tt.get_program_id x : i32
    %1 = tt.splat %0 : i32 -> tensor<1x1xi32>
    %2 = arith.cmpi slt, %1, %cst_3 : tensor<1x1xi32>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %5 = arith.cmpi slt, %4, %cst_2 : tensor<1x16xi32>
    %6 = arith.muli %1, %cst_3 : tensor<1x1xi32>
    %7 = tt.broadcast %6 : tensor<1x1xi32> -> tensor<1x16xi32>
    %8 = arith.addi %4, %7 : tensor<1x16xi32>
    %9 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %11 = tt.broadcast %2 : tensor<1x1xi1> -> tensor<1x16xi1>
    %12 = arith.andi %5, %11 : tensor<1x16xi1>
    %13 = tt.load %10, %12, %cst_1 : tensor<1x16x!tt.ptr<f32>>
    %14 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %16 = tt.load %15, %5, %cst_1 evictionPolicy = evict_last : tensor<1x16x!tt.ptr<f32>>
    %17 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %18 = tt.addptr %17, %4 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %19 = tt.load %18, %5, %cst_1 evictionPolicy = evict_last : tensor<1x16x!tt.ptr<f32>>
    %20 = tt.broadcast %2 : tensor<1x1xi1> -> tensor<1x16xi1>
    %21 = arith.andi %5, %20 : tensor<1x16xi1>
    %22 = arith.select %21, %13, %cst_1 : tensor<1x16xi1>, tensor<1x16xf32>
    %23 = tt.call @sum__fp32S1_16S__1cconstexpr_1__2cconstexpr_False_(%22) : (tensor<1x16xf32>) -> tensor<1xf32>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    %25 = arith.divf %24, %cst_0 : tensor<1x1xf32>
    %26 = tt.broadcast %25 : tensor<1x1xf32> -> tensor<1x16xf32>
    %27 = arith.subf %13, %26 : tensor<1x16xf32>
    %28 = arith.mulf %27, %27 : tensor<1x16xf32>
    %29 = tt.broadcast %2 : tensor<1x1xi1> -> tensor<1x16xi1>
    %30 = arith.andi %5, %29 : tensor<1x16xi1>
    %31 = arith.select %30, %28, %cst_1 : tensor<1x16xi1>, tensor<1x16xf32>
    %32 = tt.call @sum__fp32S1_16S__1cconstexpr_1__2cconstexpr_False_(%31) : (tensor<1x16xf32>) -> tensor<1xf32>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    %34 = arith.divf %33, %cst_0 : tensor<1x1xf32>
    %35 = arith.addf %34, %cst : tensor<1x1xf32>
    %36 = tt.extern_elementwise %35 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %37 = tt.broadcast %25 : tensor<1x1xf32> -> tensor<1x16xf32>
    %38 = arith.subf %13, %37 : tensor<1x16xf32>
    %39 = tt.broadcast %36 : tensor<1x1xf32> -> tensor<1x16xf32>
    %40 = arith.mulf %38, %39 : tensor<1x16xf32>
    %41 = arith.mulf %40, %16 : tensor<1x16xf32>
    %42 = arith.addf %41, %19 : tensor<1x16xf32>
    %43 = arith.cmpf ogt, %42, %cst_1 : tensor<1x16xf32>
    %44 = tt.extern_elementwise %42 {libname = "", libpath = "", pure = true, symbol = "__nv_expm1f"} : (tensor<1x16xf32>) -> tensor<1x16xf32>
    %45 = arith.select %43, %42, %44 : tensor<1x16xi1>, tensor<1x16xf32>
    gpu.barrier
    %46 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
    %47 = tt.addptr %46, %1 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
    tt.store %47, %36, %2 : tensor<1x1x!tt.ptr<f32>>
    %48 = arith.muli %1, %cst_3 : tensor<1x1xi32>
    %49 = tt.broadcast %48 : tensor<1x1xi32> -> tensor<1x16xi32>
    %50 = arith.addi %4, %49 : tensor<1x16xi32>
    %51 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %52 = tt.addptr %51, %50 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %53 = tt.broadcast %2 : tensor<1x1xi1> -> tensor<1x16xi1>
    %54 = arith.andi %5, %53 : tensor<1x16xi1>
    tt.store %52, %45, %54 : tensor<1x16x!tt.ptr<f32>>
    %55 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
    %56 = tt.addptr %55, %1 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
    tt.store %56, %25, %2 : tensor<1x1x!tt.ptr<f32>>
    tt.return
  }
  tt.func private @sum__fp32S1_16S__1cconstexpr_1__2cconstexpr_False_(%arg0: tensor<1x16xf32>) -> tensor<1xf32> attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = tt.call @_sum_combine__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %1 : f32
    }) : (tensor<1x16xf32>) -> tensor<1xf32>
    tt.return %0 : tensor<1xf32>
  }
  tt.func private @_sum_combine__fp32_fp32__(%arg0: f32, %arg1: f32) -> f32 attributes {noinline = false} {
    %0 = arith.addf %arg0, %arg1 : f32
    tt.return %0 : f32
  }
}