// -----// IR Dump After TritonToMaia (triton-to-maia) ('builtin.module' operation) //----- //
module {
  tt.func public @_layer_norm_bwd_dx_fused(%arg0: !tt.ptr<bf16> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8192xbf16>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8192xf32>
    %c0_i32 = arith.constant 0 : i32
    %c96_i32 = arith.constant 96 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
    %2 = tt.splat %arg10 : i32 -> tensor<8192xi32>
    %3 = arith.cmpi slt, %1, %2 : tensor<8192xi32>
    %4 = arith.muli %0, %arg9 : i32
    %5 = tt.addptr %arg4, %4 : !tt.ptr<bf16>, i32
    %6 = tt.addptr %arg1, %4 : !tt.ptr<bf16>, i32
    %7 = tt.addptr %arg0, %4 : !tt.ptr<bf16>, i32
    %8 = arith.remsi %0, %c96_i32 : i32
    %9 = tt.addptr %arg8, %8 : !tt.ptr<i32>, i32
    %10 = tt.addptr %9, %c96_i32 : !tt.ptr<i32>, i32
    %11 = arith.muli %8, %arg10 : i32
    %12 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i32
    %13 = tt.splat %12 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %15 = tt.addptr %arg3, %11 : !tt.ptr<bf16>, i32
    %16 = tt.splat %15 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %17 = tt.addptr %16, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %18 = tt.splat %5 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %19 = tt.addptr %18, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %20 = tt.load %19, %3, %cst : tensor<8192x!tt.ptr<bf16>>
    %21 = arith.extf %20 : tensor<8192xbf16> to tensor<8192xf32>
    %22 = tt.splat %6 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %23 = tt.addptr %22, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %24 = tt.load %23, %3, %cst : tensor<8192x!tt.ptr<bf16>>
    %25 = arith.extf %24 : tensor<8192xbf16> to tensor<8192xf32>
    %26 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %27 = tt.addptr %26, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %28 = tt.load %27, %3 : tensor<8192x!tt.ptr<bf16>>
    %29 = arith.extf %28 : tensor<8192xbf16> to tensor<8192xf32>
    %30 = tt.addptr %arg6, %0 : !tt.ptr<f32>, i32
    %31 = tt.load %30 : !tt.ptr<f32>
    %32 = tt.addptr %arg7, %0 : !tt.ptr<f32>, i32
    %33 = tt.load %32 : !tt.ptr<f32>
    %34 = tt.splat %31 : f32 -> tensor<8192xf32>
    %35 = arith.subf %21, %34 : tensor<8192xf32>
    %36 = tt.splat %33 : f32 -> tensor<8192xf32>
    %37 = arith.mulf %35, %36 : tensor<8192xf32>
    %38 = arith.mulf %29, %25 : tensor<8192xf32>
    %39 = arith.select %3, %37, %cst_0 : tensor<8192xi1>, tensor<8192xf32>
    %40 = arith.select %3, %38, %cst_0 : tensor<8192xi1>, tensor<8192xf32>
    %41 = arith.mulf %39, %40 : tensor<8192xf32>
    %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
    ^bb0(%arg11: f32, %arg12: f32):
      %63 = arith.addf %arg11, %arg12 : f32
      tt.reduce.return %63 : f32
    }) : (tensor<8192xf32>) -> f32
    %43 = arith.sitofp %arg10 : i32 to f32
    %44 = arith.divf %42, %43 : f32
    %45 = "tt.reduce"(%40) <{axis = 0 : i32}> ({
    ^bb0(%arg11: f32, %arg12: f32):
      %63 = arith.addf %arg11, %arg12 : f32
      tt.reduce.return %63 : f32
    }) : (tensor<8192xf32>) -> f32
    %46 = arith.divf %45, %43 : f32
    %47 = tt.splat %44 : f32 -> tensor<8192xf32>
    %48 = arith.mulf %39, %47 : tensor<8192xf32>
    %49 = tt.splat %46 : f32 -> tensor<8192xf32>
    %50 = arith.addf %48, %49 : tensor<8192xf32>
    %51 = arith.subf %40, %50 : tensor<8192xf32>
    %52 = arith.mulf %51, %36 : tensor<8192xf32>
    %53 = tt.splat %7 : !tt.ptr<bf16> -> tensor<8192x!tt.ptr<bf16>>
    %54 = tt.addptr %53, %1 : tensor<8192x!tt.ptr<bf16>>, tensor<8192xi32>
    %55 = arith.truncf %52 : tensor<8192xf32> to tensor<8192xbf16>
    tt.store %54, %55, %3 : tensor<8192x!tt.ptr<bf16>>
    %56 = arith.mulf %25, %39 : tensor<8192xf32>
    scf.while : () -> () {
      %63 = tt.atomic_cas acq_rel, gpu, %9, %c0_i32, %c1_i32 : (!tt.ptr<i32>, i32, i32) -> i32
      %64 = arith.cmpi eq, %63, %c1_i32 : i32
      scf.condition(%64)
    } do {
      scf.yield
    }
    %57 = tt.load %10 : !tt.ptr<i32>
    %58 = arith.cmpi eq, %57, %c0_i32 : i32
    %59:2 = scf.if %58 -> (tensor<8192xf32>, tensor<8192xf32>) {
      %63 = tt.atomic_rmw exch, acq_rel, gpu, %10, %c1_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %56, %25 : tensor<8192xf32>, tensor<8192xf32>
    } else {
      %63 = tt.load %14, %3 : tensor<8192x!tt.ptr<bf16>>
      %64 = arith.extf %63 : tensor<8192xbf16> to tensor<8192xf32>
      %65 = arith.addf %56, %64 : tensor<8192xf32>
      %66 = tt.load %17, %3 : tensor<8192x!tt.ptr<bf16>>
      %67 = arith.extf %66 : tensor<8192xbf16> to tensor<8192xf32>
      %68 = arith.addf %25, %67 : tensor<8192xf32>
      scf.yield %65, %68 : tensor<8192xf32>, tensor<8192xf32>
    }
    %60 = arith.truncf %59#0 : tensor<8192xf32> to tensor<8192xbf16>
    tt.store %14, %60, %3 : tensor<8192x!tt.ptr<bf16>>
    %61 = arith.truncf %59#1 : tensor<8192xf32> to tensor<8192xbf16>
    tt.store %17, %61, %3 : tensor<8192x!tt.ptr<bf16>>
    %62 = tt.atomic_rmw exch, acq_rel, gpu, %9, %c0_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}
