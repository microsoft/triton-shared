module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c6 = arith.constant 6 : index
    %c6144 = arith.constant 6144 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%c6144], sizes: [256, 128], strides: [1, %c6] : memref<*xbf16> to memref<256x128xbf16, strided<[1, ?], offset: 512>>
    %alloc = memref.alloc() : memref<256x128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<256x128xbf16, strided<[1, ?], offset: 512>> to memref<256x128xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
    bufferization.materialize_in_destination %0 in writable %reinterpret_cast : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 512>>) -> ()
    return
  }
}

