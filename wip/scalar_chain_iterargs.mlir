module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg7 : i32 to index
    %1 = tt.addptr %arg1, %arg7 : !tt.ptr<f32, 1>, i32
    %2 = arith.sitofp %arg7 : i32 to f32
    %3:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %1, %arg12 = %0) -> (!tt.ptr<f32, 1>, index)  : i32 {
      tt.store %arg11, %2 {cache = 1 : i32, evict = 1 : i32} : f32
      %4 = tt.addptr %arg11, %arg10 : !tt.ptr<f32, 1>, i32
      %5 = arith.index_cast %arg10 : i32 to index
      %6 = arith.addi %arg12, %5 : index
      scf.yield %4, %6 : !tt.ptr<f32, 1>, index
    }
    return
  }
}

// this crashes in --triton-to-linalg
