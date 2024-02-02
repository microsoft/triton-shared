module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c3072 = arith.constant 3072 : index
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c259 = arith.constant 259 : index
    %c130 = arith.constant 130 : index
    %cst = arith.constant 0xFF80 : bf16
    %0 = arith.addi %c2, %c3072 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [128, 256], strides: [1, %c1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
    %1 = arith.addi %c2, %c3072 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [128, 256], strides: [1, %c1024] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.minsi %2, %c130 : index
    %4 = arith.subi %3, %c2 : index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.minsi %5, %c259 : index
    %7 = arith.subi %6, %c3 : index
    %8 = arith.minsi %4, %c128 : index
    %9 = arith.minsi %7, %c256 : index
    %alloc = memref.alloc() : memref<128x256xbf16>
    %false = arith.constant false
    %c128_1 = arith.constant 128 : index
    %10 = arith.cmpi slt, %8, %c128_1 : index
    %11 = arith.ori %false, %10 : i1
    %c256_2 = arith.constant 256 : index
    %12 = arith.cmpi slt, %9, %c256_2 : index
    %13 = arith.ori %11, %12 : i1
    scf.if %13 {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x256xbf16>)
    }
    memref.copy %reinterpret_cast, %alloc : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<128x256xbf16>
    %14 = bufferization.to_tensor %alloc restrict writable : memref<128x256xbf16>
    %15 = arith.index_cast %arg2 : i32 to index
    %16 = arith.minsi %15, %c130 : index
    %17 = arith.subi %16, %c2 : index
    %18 = arith.index_cast %arg3 : i32 to index
    %19 = arith.minsi %18, %c259 : index
    %20 = arith.subi %19, %c3 : index
    %21 = arith.minsi %17, %c128 : index
    %22 = arith.minsi %20, %c256 : index
    %extracted_slice = tensor.extract_slice %14[0, 0] [%21, %22] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
    %subview = memref.subview %reinterpret_cast_0[0, 0] [%21, %22] [1, 1] : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[1, ?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}

