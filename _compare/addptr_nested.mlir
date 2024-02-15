#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c15 = arith.constant 15 : index
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [4, 256], strides: [1, %c5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.addi %2, %3 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [4, 256], strides: [2, %c10] : memref<*xbf16> to memref<4x256xbf16, strided<[2, ?], offset: ?>>
    %alloc_1 = memref.alloc() : memref<4x256xbf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<4x256xbf16, strided<[2, ?], offset: ?>> to memref<4x256xbf16>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x256xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%1 : tensor<4x256xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %12 = arith.addf %in, %in_3 : bf16
      linalg.yield %12 : bf16
    } -> tensor<4x256xbf16>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.addi %7, %8 : index
    %10 = arith.index_cast %arg1 : i32 to index
    %11 = arith.addi %9, %10 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [4, 256], strides: [3, %c15] : memref<*xbf16> to memref<4x256xbf16, strided<[3, ?], offset: ?>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[3, ?], offset: ?>>) -> ()
    return
  }
}

