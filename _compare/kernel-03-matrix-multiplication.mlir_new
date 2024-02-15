#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel_0123456789101112131415(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c255_i32 : i32
    %5 = arith.divsi %4, %c256_i32 : i32
    %6 = arith.addi %arg5, %c63_i32 : i32
    %7 = arith.divsi %6, %c64_i32 : i32
    %8 = arith.muli %5, %c8_i32 : i32
    %9 = arith.divsi %arg15, %8 : i32
    %10 = arith.muli %9, %c8_i32 : i32
    %11 = arith.subi %3, %10 : i32
    %12 = arith.minsi %11, %c8_i32 : i32
    %13 = arith.remsi %arg15, %12 : i32
    %14 = arith.addi %10, %13 : i32
    %15 = arith.remsi %arg15, %8 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %17 : i32 to index
    %20 = arith.muli %16, %c256_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.index_cast %20 : i32 to index
    %23 = arith.index_cast %arg6 : i32 to index
    %24 = arith.muli %19, %23 : index
    %25 = arith.index_cast %arg7 : i32 to index
    %26 = arith.index_cast %arg8 : i32 to index
    %27 = arith.index_cast %arg9 : i32 to index
    %28 = arith.muli %22, %27 : index
    %29 = arith.muli %arg7, %c64_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %arg8, %c64_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33:3 = scf.for %arg18 = %c0_i32 to %7 step %c1_i32 iter_args(%arg19 = %1, %arg20 = %24, %arg21 = %c0) -> (tensor<128x256xf32>, index, index)  : i32 {
      %53 = arith.addi %arg21, %28 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%53], sizes: [64, 256], strides: [%26, %27] : memref<*xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%arg20], sizes: [128, 64], strides: [%23, %25] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<128x64xbf16>
      memref.copy %reinterpret_cast_1, %alloc : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
      %54 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
      %alloc_2 = memref.alloc() : memref<64x256xbf16>
      memref.copy %reinterpret_cast_0, %alloc_2 : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<64x256xbf16>
      %55 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x256xbf16>
      %56 = tensor.empty() : tensor<128x256xf32>
      %57 = linalg.fill ins(%cst : f32) outs(%56 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %58 = linalg.matmul ins(%54, %55 : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%57 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %59 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg19, %58 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%arg19 : tensor<128x256xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %62 = arith.addf %in, %in_3 : f32
        linalg.yield %62 : f32
      } -> tensor<128x256xf32>
      %60 = arith.addi %arg20, %30 : index
      %61 = arith.addi %arg21, %32 : index
      scf.yield %59, %60, %61 : tensor<128x256xf32>, index, index
    }
    %34 = tensor.empty() : tensor<128x256xbf16>
    %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%33#0 : tensor<128x256xf32>) outs(%34 : tensor<128x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %53 = arith.truncf %in : f32 to bf16
      linalg.yield %53 : bf16
    } -> tensor<128x256xbf16>
    %36 = arith.index_cast %arg10 : i32 to index
    %37 = arith.muli %18, %36 : index
    %38 = arith.index_cast %arg11 : i32 to index
    %39 = arith.muli %21, %38 : index
    %40 = arith.addi %37, %39 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%40], sizes: [128, 256], strides: [%36, %38] : memref<*xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>
    %41 = arith.index_cast %17 : i32 to index
    %42 = arith.addi %41, %c128 : index
    %43 = arith.index_cast %arg3 : i32 to index
    %44 = arith.minsi %42, %43 : index
    %45 = arith.subi %44, %41 : index
    %46 = arith.index_cast %20 : i32 to index
    %47 = arith.addi %46, %c256 : index
    %48 = arith.index_cast %arg4 : i32 to index
    %49 = arith.minsi %47, %48 : index
    %50 = arith.subi %49, %46 : index
    %51 = arith.minsi %45, %c128 : index
    %52 = arith.minsi %50, %c256 : index
    %extracted_slice = tensor.extract_slice %35[0, 0] [%51, %52] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%51, %52] [1, 1] : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

