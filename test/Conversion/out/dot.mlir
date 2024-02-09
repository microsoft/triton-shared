#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128, 64], strides: [%c128, 1] : memref<*xbf16> to memref<128x64xbf16, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<128x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<128x64xbf16, strided<[?, 1]>> to memref<128x64xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256, 64], strides: [1, %c256] : memref<*xbf16> to memref<256x64xbf16, strided<[1, ?]>>
    %alloc_1 = memref.alloc() : memref<256x64xbf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<256x64xbf16, strided<[1, ?]>> to memref<256x64xbf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<256x64xbf16>
    %2 = tensor.empty() : tensor<64x256xbf16>
    %transposed = linalg.transpose ins(%1 : tensor<256x64xbf16>) outs(%2 : tensor<64x256xbf16>) permutation = [1, 0] 
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 256], strides: [%c256, 1] : memref<*xbf16> to memref<128x256xbf16, strided<[?, 1]>>
    %alloc_3 = memref.alloc() : memref<128x256xbf16>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<128x256xbf16, strided<[?, 1]>> to memref<128x256xbf16>
    %3 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x256xbf16>
    %4 = tensor.empty() : tensor<128x256xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%4 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %6 = linalg.matmul ins(%0, %transposed : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%5 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %3 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
      %8 = arith.addf %in, %in_4 : bf16
      linalg.yield %8 : bf16
    } -> tensor<128x256xbf16>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_2 : (tensor<128x256xbf16>, memref<128x256xbf16, strided<[?, 1]>>) -> ()
    return
  }
}

