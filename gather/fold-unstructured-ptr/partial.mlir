module {
  func.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %arg4 : !tt.ptr<f32>, i32
    %1 = builtin.unrealized_conversion_cast %0 : !tt.ptr<f32> to memref<1xf32, strided<[1], offset: ?>>
    %2 = scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg14 = %1) -> (memref<1xf32, strided<[1], offset: ?>>)  : i32 {
      %3 = scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg16 = %arg14) -> (memref<1xf32, strided<[1], offset: ?>>)  : i32 {
        %4 = builtin.unrealized_conversion_cast %arg16 : memref<1xf32, strided<[1], offset: ?>> to !tt.ptr<f32>
        %5 = arith.muli %arg13, %arg15 : i32
        %6 = arith.sitofp %5 : i32 to f32
        tt.store %4, %6 : !tt.ptr<f32>
        %7 = tt.addptr %4, %c1_i32 : !tt.ptr<f32>, i32
        %8 = builtin.unrealized_conversion_cast %7 : !tt.ptr<f32> to memref<1xf32, strided<[1], offset: ?>>
        scf.yield %8 : memref<1xf32, strided<[1], offset: ?>>
      }
      scf.yield %3 : memref<1xf32, strided<[1], offset: ?>>
    }
    return
  }
}
