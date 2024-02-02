module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c11 = arith.constant 11 : index
    %c20480 = arith.constant 20480 : index
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.addi %1, %c20480 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1024], strides: [%c11] : memref<*xbf16> to memref<1024xbf16, strided<[?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024xbf16, strided<[?], offset: ?>> to memref<1024xbf16>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<1024xbf16>, memref<1024xbf16, strided<[1], offset: ?>>) -> ()
    return
  }
}

