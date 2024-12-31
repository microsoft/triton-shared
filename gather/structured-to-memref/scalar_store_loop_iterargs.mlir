module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.index_cast %arg7 : i32 to index
    %1 = arith.index_cast %arg7 : i32 to index
    %2 = arith.sitofp %arg7 : i32 to f32
    %3:3 = scf.for %arg16 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg17 = %arg7, %arg18 = %1, %arg19 = %0) -> (i32, index, index)  : i32 {
      %4 = "tts.make_unstructured_tptr"(%arg1, %arg17) : (!tt.ptr<f32>, i32) -> !tt.ptr<f32>
      tt.store %4, %2 : !tt.ptr<f32>
      %5 = arith.index_cast %arg16 : i32 to index
      %6 = arith.addi %arg18, %5 : index
      %7 = arith.addi %arg17, %arg16 : i32
      %8 = arith.index_cast %arg16 : i32 to index
      %9 = arith.addi %arg19, %8 : index
      scf.yield %7, %6, %9 : i32, index, index
    }
    return
  }
}