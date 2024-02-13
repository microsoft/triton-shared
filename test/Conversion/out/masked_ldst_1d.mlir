module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0xFF80 : bf16
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.minsi %0, %c128 : index
    %alloc = memref.alloc() : memref<128xbf16>
    %2 = arith.cmpi slt, %1, %c128 : index
    scf.if %2 {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<128xbf16>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%1] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
    %subview_1 = memref.subview %alloc[0] [%1] [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.minsi %4, %c128 : index
    %extracted_slice = tensor.extract_slice %3[0] [%5] [1] : tensor<128xbf16> to tensor<?xbf16>
    %subview_2 = memref.subview %reinterpret_cast_0[0] [%5] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xbf16>, memref<?xbf16, strided<[1]>>) -> ()
    return
  }
}

