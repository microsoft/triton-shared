module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg7 = %c0 to %c12 step %c3 iter_args(%arg8 = %c1024) -> (index) {
      %1 = arith.addi %arg8, %c3 : index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [%c1] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
      %alloc = memref.alloc() : memref<256xbf16>
      memref.copy %reinterpret_cast, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
      %2 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16>
      bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
      scf.yield %1 : index
    }
    return
  }
}

