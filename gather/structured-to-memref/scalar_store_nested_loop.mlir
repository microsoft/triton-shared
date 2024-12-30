module {
  func.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %arg4 : !tt.ptr<f32>, i32
    %1 = scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg14 = %0) -> (!tt.ptr<f32>)  : i32 {
      %2 = scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg16 = %arg14) -> (!tt.ptr<f32>)  : i32 {
        %3 = arith.muli %arg13, %arg15 : i32
        %4 = arith.sitofp %3 : i32 to f32
        tt.store %arg16, %4 : !tt.ptr<f32>
        %5 = tt.addptr %arg16, %c1_i32 : !tt.ptr<f32>, i32
        scf.yield %5 : !tt.ptr<f32>
      }
      scf.yield %2 : !tt.ptr<f32>
    }
    return
  }
}