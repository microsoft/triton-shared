#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel_with_block_pointers_01234567891011(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x64xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = arith.index_cast %arg12 : i32 to index
    %3 = arith.index_cast %arg6 : i32 to index
    %4 = arith.index_cast %arg7 : i32 to index
    %5 = arith.muli %2, %3 : index
    %6 = arith.muli %4, %c64 : index
    %7 = arith.addi %5, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
    %8:7 = scf.for %arg20 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg21 = %1, %arg22 = %reinterpret_cast, %arg23 = %reinterpret_cast_0, %arg24 = %7, %arg25 = %c0, %arg26 = %5, %arg27 = %c0) -> (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc = memref.alloc() : memref<128x64xbf16>
      memref.copy %arg22, %alloc : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
      %17 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
      %alloc_2 = memref.alloc() : memref<128x64xbf16>
      memref.copy %arg23, %alloc_2 : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
      %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<128x64xbf16>
      %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %18 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%17 : tensor<128x64xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %27 = arith.addf %in, %in_5 : bf16
        linalg.yield %27 : bf16
      } -> tensor<128x64xbf16>
      %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg21, %19 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%arg21 : tensor<128x64xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %27 = arith.addf %in, %in_5 : bf16
        linalg.yield %27 : bf16
      } -> tensor<128x64xbf16>
      %21 = arith.muli %4, %c64 : index
      %22 = arith.addi %21, %arg25 : index
      %23 = arith.addi %arg24, %22 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
      %24 = arith.muli %3, %c64 : index
      %25 = arith.addi %24, %arg26 : index
      %26 = arith.addi %25, %arg27 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
      scf.yield %20, %reinterpret_cast_3, %reinterpret_cast_4, %23, %c0, %26, %c0 : tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %9 = arith.muli %arg13, %c256_i32 : i32
    %10 = arith.index_cast %arg12 : i32 to index
    %11 = arith.index_cast %9 : i32 to index
    %12 = arith.index_cast %arg10 : i32 to index
    %13 = arith.index_cast %arg11 : i32 to index
    %14 = arith.muli %10, %12 : index
    %15 = arith.muli %11, %13 : index
    %16 = arith.addi %14, %15 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [128, 64], strides: [%12, %13] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %8#0 in writable %reinterpret_cast_1 : (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

