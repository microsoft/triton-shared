#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c256_i32 = arith.constant 256 : i32
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c256 = arith.constant 256 : index
    %0 = tensor.empty() : tensor<4x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x256xbf16>) -> tensor<4x256xbf16>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<1x256xbf16, strided<[256, 1]>>
    %alloc = memref.alloc() : memref<1x256xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<1x256xbf16, strided<[256, 1]>> to memref<1x256xbf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x256xbf16>
    %3 = arith.index_cast %arg1 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<1x256xbf16>, memref<1x256xbf16, strided<[256, 1], offset: ?>>) -> ()
    %4:2 = scf.for %arg8 = %c0 to %c12 step %c3 iter_args(%arg9 = %1, %arg10 = %c0) -> (tensor<4x256xbf16>, index) {
      %5 = arith.index_cast %arg8 : index to i32
      %6 = arith.muli %5, %c256_i32 : i32
      %7 = arith.index_cast %6 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg10], sizes: [4, 256], strides: [%7, %c1] : memref<*xbf16> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
      %alloc_3 = memref.alloc() : memref<4x256xbf16>
      memref.copy %reinterpret_cast_2, %alloc_3 : memref<4x256xbf16, strided<[?, ?], offset: ?>> to memref<4x256xbf16>
      %8 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x256xbf16>
      %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg9, %8 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%arg9 : tensor<4x256xbf16>) {
      ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
        %11 = arith.addf %in, %in_4 : bf16
        linalg.yield %11 : bf16
      } -> tensor<4x256xbf16>
      %10 = arith.addi %arg10, %c256 : index
      scf.yield %9, %10 : tensor<4x256xbf16>, index
    }
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 256], strides: [%c256, 1] : memref<*xbf16> to memref<4x256xbf16, strided<[?, 1]>>
    bufferization.materialize_in_destination %4#0 in writable %reinterpret_cast_1 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[?, 1]>>) -> ()
    return
  }
}

