#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [4, 256], strides: [1, %c5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16>
    %2 = arith.index_cast %arg3 : i32 to index
    %3:2 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %2) -> (tensor<4x256xbf16>, index) {
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%arg13], sizes: [4, 256], strides: [%c1, %c5] : memref<*xbf16> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
      %alloc_2 = memref.alloc() : memref<4x256xbf16>
      memref.copy %reinterpret_cast_1, %alloc_2 : memref<4x256xbf16, strided<[?, ?], offset: ?>> to memref<4x256xbf16>
      %5 = bufferization.to_tensor %alloc_2 restrict writable : memref<4x256xbf16>
      %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %5 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%arg12 : tensor<4x256xbf16>) {
      ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
        %8 = arith.addf %in, %in_3 : bf16
        linalg.yield %8 : bf16
      } -> tensor<4x256xbf16>
      %7 = arith.addi %arg13, %c3 : index
      scf.yield %6, %7 : tensor<4x256xbf16>, index
    }
    %4 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [4, 256], strides: [1, %c5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
    bufferization.materialize_in_destination %3#0 in writable %reinterpret_cast_0 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, ?], offset: ?>>) -> ()
    return
  }
}

