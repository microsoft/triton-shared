module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.muli %1, %c2048 : index
    %3 = arith.addi %0, %2 : index
    %4 = arith.addi %1, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [%4] : memref<*xbf16> to memref<1024xbf16, strided<[?], offset: ?>>
    %5 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024xbf16, strided<[?], offset: ?>> to memref<1024xbf16>
    %6 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_0 : (tensor<1024xbf16>, memref<1024xbf16, strided<[1], offset: ?>>) -> ()
    return
  }
}

