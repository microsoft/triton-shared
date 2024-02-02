module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 7.000000e+00 : bf16
    %c128 = arith.constant 128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.minsi %0, %c128 : index
    %alloc = memref.alloc() : memref<128xbf16>
    %false = arith.constant false
    %c128_1 = arith.constant 128 : index
    %2 = arith.cmpi slt, %1, %c128_1 : index
    %3 = arith.ori %false, %2 : i1
    scf.if %3 {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<128xbf16>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%1] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
    %subview_2 = memref.subview %alloc[0] [%1] [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<?xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.minsi %5, %c128 : index
    return
  }
}

