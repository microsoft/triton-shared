module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c6 = arith.constant 6 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [4, 256], strides: [1, %c6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %1 = arith.index_cast %arg2 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [4, 256], strides: [1, %c6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}

