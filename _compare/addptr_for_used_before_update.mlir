module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c1024], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
    %0:2 = scf.for %arg7 = %c0 to %c12 step %c3 iter_args(%arg8 = %reinterpret_cast, %arg9 = %c1024) -> (memref<256xbf16, strided<[?], offset: ?>>, index) {
      %alloc = memref.alloc() : memref<256xbf16>
      memref.copy %arg8, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
      %1 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16>
      bufferization.materialize_in_destination %1 in writable %arg8 : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
      %2 = arith.addi %arg9, %c3 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      scf.yield %reinterpret_cast_0, %2 : memref<256xbf16, strided<[?], offset: ?>>, index
    }
    return
  }
}

