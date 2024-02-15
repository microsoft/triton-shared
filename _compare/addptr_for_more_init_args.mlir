module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c1024], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c1024], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
    %0:7 = scf.for %arg8 = %c0 to %c12 step %c3 iter_args(%arg9 = %c1, %arg10 = %reinterpret_cast, %arg11 = %c2, %arg12 = %reinterpret_cast_0, %arg13 = %c3, %arg14 = %c1024, %arg15 = %c1024) -> (index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index) {
      %alloc = memref.alloc() : memref<256xbf16>
      memref.copy %arg10, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
      %1 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16>
      bufferization.materialize_in_destination %1 in writable %arg12 : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
      %2 = arith.addi %arg14, %c3 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      %3 = arith.addi %arg9, %c3 : index
      %4 = arith.addi %arg11, %c3 : index
      %5 = arith.addi %arg13, %c3 : index
      %6 = arith.addi %3, %4 : index
      %7 = arith.addi %6, %5 : index
      %8 = arith.addi %arg15, %7 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%8], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      scf.yield %3, %reinterpret_cast_1, %4, %reinterpret_cast_2, %5, %2, %8 : index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index
    }
    return
  }
}

