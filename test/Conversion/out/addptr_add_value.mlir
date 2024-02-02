module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c10 = arith.constant 10 : index
    %c6 = arith.constant 6 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.addi %0, %1 : index
    %3 = arith.addi %2, %c10 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [4, 256], strides: [1, %c6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.addi %4, %5 : index
    %7 = arith.addi %6, %c10 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [4, 256], strides: [1, %c6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16>
    bufferization.materialize_in_destination %8 in writable %reinterpret_cast_0 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}

