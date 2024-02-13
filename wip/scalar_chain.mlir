module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.addptr %arg1, %arg7 : !tt.ptr<f32, 1>, i32
    %1 = arith.sitofp %arg7 : i32 to f32
    scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %2 = tt.addptr %0, %arg10 : !tt.ptr<f32, 1>, i32
      tt.store %2, %1 {cache = 1 : i32, evict = 1 : i32} : f32
    }
    return
  }
}


// module {
//   func.func @reduce_kernel_2d_0d1d2de3de(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) {
//     %c1_i32 = arith.constant 1 : i32
//     %c5_i32 = arith.constant 5 : i32
//     %c0_i32 = arith.constant 0 : i32
//     %0 = arith.sitofp %arg7 : i32 to f32
//     scf.for %arg16 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
//       %1 = arith.index_cast %arg7 : i32 to index
//       %2 = arith.index_cast %arg16 : i32 to index
//       %3 = arith.addi %1, %2 : index
//       %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
//       affine.store %0, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
//     }
//     return
//   }
// }