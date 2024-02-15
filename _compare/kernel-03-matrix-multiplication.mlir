#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel_0123456789101112131415(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
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
    %18 = arith.muli %16, %c256_i32 : i32
    %19 = arith.index_cast %17 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.muli %19, %20 : index
    %22 = arith.index_cast %arg7 : i32 to index
    %23 = arith.index_cast %arg8 : i32 to index
    %24 = arith.index_cast %18 : i32 to index
    %25 = arith.index_cast %arg9 : i32 to index
    %26 = arith.muli %24, %25 : index
    %27 = arith.muli %arg7, %c64_i32 : i32
    %28 = arith.muli %arg8, %c64_i32 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [128, 64], strides: [%20, %22] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 256], strides: [%23, %25] : memref<*xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
    %29:7 = scf.for %arg18 = %c0_i32 to %7 step %c1_i32 iter_args(%arg19 = %1, %arg20 = %reinterpret_cast, %arg21 = %reinterpret_cast_0, %arg22 = %21, %arg23 = %c0, %arg24 = %26, %arg25 = %c0) -> (tensor<128x256xf32>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<64x256xbf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc = memref.alloc() : memref<128x64xbf16>
      memref.copy %arg20, %alloc : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
      %51 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
      %alloc_2 = memref.alloc() : memref<64x256xbf16>
      memref.copy %arg21, %alloc_2 : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<64x256xbf16>
      %52 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x256xbf16>
      %53 = tensor.empty() : tensor<128x256xf32>
      %54 = linalg.fill ins(%cst : f32) outs(%53 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %55 = linalg.matmul ins(%51, %52 : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%54 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %56 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%55, %1 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%55 : tensor<128x256xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %64 = arith.addf %in, %in_5 : f32
        linalg.yield %64 : f32
      } -> tensor<128x256xf32>
      %57 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg19, %56 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%arg19 : tensor<128x256xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %64 = arith.addf %in, %in_5 : f32
        linalg.yield %64 : f32
      } -> tensor<128x256xf32>
      %58 = arith.index_cast %27 : i32 to index
      %59 = arith.addi %arg22, %58 : index
      %60 = arith.addi %59, %arg23 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%60], sizes: [128, 64], strides: [%20, %22] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
      %61 = arith.index_cast %28 : i32 to index
      %62 = arith.addi %arg24, %61 : index
      %63 = arith.addi %62, %arg25 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%63], sizes: [64, 256], strides: [%23, %25] : memref<*xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      scf.yield %57, %reinterpret_cast_3, %reinterpret_cast_4, %60, %c0, %63, %c0 : tensor<128x256xf32>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<64x256xbf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %30 = tensor.empty() : tensor<128x256xbf16>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%29#0 : tensor<128x256xf32>) outs(%30 : tensor<128x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %51 = arith.truncf %in : f32 to bf16
      linalg.yield %51 : bf16
    } -> tensor<128x256xbf16>
    %32 = arith.index_cast %arg10 : i32 to index
    %33 = arith.index_cast %17 : i32 to index
    %34 = arith.muli %33, %32 : index
    %35 = arith.index_cast %arg11 : i32 to index
    %36 = arith.index_cast %18 : i32 to index
    %37 = arith.muli %36, %35 : index
    %38 = arith.addi %34, %37 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [128, 256], strides: [%32, %35] : memref<*xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>
    %39 = arith.index_cast %17 : i32 to index
    %40 = arith.addi %39, %c128 : index
    %41 = arith.index_cast %arg3 : i32 to index
    %42 = arith.minsi %40, %41 : index
    %43 = arith.subi %42, %39 : index
    %44 = arith.index_cast %18 : i32 to index
    %45 = arith.addi %44, %c256 : index
    %46 = arith.index_cast %arg4 : i32 to index
    %47 = arith.minsi %45, %46 : index
    %48 = arith.subi %47, %44 : index
    %49 = arith.minsi %43, %c128 : index
    %50 = arith.minsi %48, %c256 : index
    %extracted_slice = tensor.extract_slice %31[0, 0] [%49, %50] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
    %subview = memref.subview %reinterpret_cast_1[0, 0] [%49, %50] [1, 1] : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

