module {
  func.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %arg4 : !tt.ptr<f32, 1>, i32
    %1 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %0) -> (!tt.ptr<f32, 1>)  : i32 {
      %2 = arith.sitofp %arg7 : i32 to f32
      %3 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %arg8) -> (!tt.ptr<f32, 1>)  : i32 {
        tt.store %arg10, %2 {cache = 1 : i32, evict = 1 : i32} : f32
        %4 = tt.addptr %arg10, %c1_i32 : !tt.ptr<f32, 1>, i32
        scf.yield %4 : !tt.ptr<f32, 1>
      }
      scf.yield %3 : !tt.ptr<f32, 1>
    }
    return
  }
}

