module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0xFF80 : bf16
    %c130 = arith.constant 130 : index
    %c2 = arith.constant 2 : index
    %c259 = arith.constant 259 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c3074 = arith.constant 3074 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c3074], sizes: [128, 256], strides: [1, %c1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c3074], sizes: [128, 256], strides: [1, %c1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.minsi %0, %c130 : index
    %2 = arith.subi %1, %c2 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.minsi %3, %c259 : index
    %5 = arith.subi %4, %c3 : index
    %6 = arith.minsi %2, %c128 : index
    %7 = arith.minsi %5, %c256 : index
    %alloc = memref.alloc() : memref<128x256xbf16>
    %8 = arith.cmpi slt, %6, %c128 : index
    %9 = arith.cmpi slt, %7, %c256 : index
    %10 = arith.ori %8, %9 : i1
    scf.if %10 {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x256xbf16>)
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%6, %7] [1, 1] : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[1, ?], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%6, %7] [1, 1] : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<128x256xbf16>
    %12 = arith.index_cast %arg2 : i32 to index
    %13 = arith.minsi %12, %c130 : index
    %14 = arith.subi %13, %c2 : index
    %15 = arith.index_cast %arg3 : i32 to index
    %16 = arith.minsi %15, %c259 : index
    %17 = arith.subi %16, %c3 : index
    %18 = arith.minsi %14, %c128 : index
    %19 = arith.minsi %17, %c256 : index
    %extracted_slice = tensor.extract_slice %11[0, 0] [%18, %19] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
    %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%18, %19] [1, 1] : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[1, ?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}

