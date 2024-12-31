module {
  tt.func public @gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_0 = arith.constant 0 : i64
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %cst_1 = arith.constant dense<64> : tensor<4xi32>
    %cst_2 = arith.constant dense<3> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %c0_i64_0 : i64 -> tensor<4xi64>
    %2 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
    %3:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0, %arg5 = %c0) -> (tensor<4xi32>, tensor<4xi32>, index)  : i32 {
      %4 = arith.divsi %arg3, %cst_2 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
      %8 = arith.extsi %6 : tensor<4xi32> to tensor<4xi64>
      %9 = arith.addi %1, %8 : tensor<4xi64>
      %10 = "tts.make_unstructured_tptr"(%arg0, %9) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      %11 = builtin.unrealized_conversion_cast %9 : tensor<4xi64> to tensor<4x!tt.ptr<f32>> {"reconvert-offset-to-ptr"}
      %12 = tt.load %10, %7 : tensor<4x!tt.ptr<f32>>
      %13 = arith.extsi %6 : tensor<4xi32> to tensor<4xi64>
      %14 = arith.addi %2, %13 : tensor<4xi64>
      %15 = "tts.make_unstructured_tptr"(%arg1, %14) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      %16 = builtin.unrealized_conversion_cast %14 : tensor<4xi64> to tensor<4x!tt.ptr<f32>> {"reconvert-offset-to-ptr"}
      tt.store %15, %12 : tensor<4x!tt.ptr<f32>>
      %17 = arith.addi %6, %cst : tensor<4xi32>
      %18 = arith.addi %arg4, %cst : tensor<4xi32>
      %19 = arith.addi %arg5, %c4 : index
      scf.yield %17, %18, %19 : tensor<4xi32>, tensor<4xi32>, index
    }
    tt.return
  }
}
