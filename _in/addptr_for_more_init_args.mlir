module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %0:5 = scf.for %arg8 = %c0 to %c12 step %c3 iter_args(%arg9 = %c1, %arg10 = %c2, %arg11 = %c3, %arg12 = %c1024, %arg13 = %c1024) -> (index, index, index, index, index) {
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%arg13], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%arg12], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      %alloc = memref.alloc() : memref<256xbf16>
      memref.copy %reinterpret_cast_0, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
      %1 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16>
      bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
      %2 = arith.addi %arg12, %c3 : index
      %3 = arith.addi %arg9, %c3 : index
      %4 = arith.addi %arg10, %c3 : index
      %5 = arith.addi %arg11, %c3 : index
      %6 = arith.addi %3, %4 : index
      %7 = arith.addi %6, %5 : index
      %8 = arith.addi %arg13, %7 : index
      scf.yield %3, %4, %5, %2, %8 : index, index, index, index, index
    }
    return
  }
}

