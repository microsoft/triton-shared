#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel_0d1d2d34567c89c1011c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = arith.addi %arg3, %c31_i32 : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.addi %arg4, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg12, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg12, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.muli %14, %c64_i32 : i32
    %17 = arith.index_cast %arg3 : i32 to index
    %18 = arith.index_cast %15 : i32 to index
    %19 = arith.index_cast %arg6 : i32 to index
    %20 = arith.muli %18, %19 : index
    %21 = arith.remsi %20, %19 : index
    %22 = arith.muli %17, %19 : index
    %23 = arith.addi %22, %21 : index
    %24 = arith.subi %23, %20 : index
    %25 = arith.divsi %24, %19 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%25, %c16], strides: [%19, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
    %26 = arith.subi %c32, %25 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%26, %c16], strides: [%19, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
    %27 = arith.index_cast %arg7 : i32 to index
    %28 = arith.index_cast %arg4 : i32 to index
    %29 = arith.index_cast %16 : i32 to index
    %30 = arith.remsi %29, %28 : index
    %31 = arith.subi %29, %30 : index
    %32 = arith.addi %30, %c64 : index
    %33 = arith.minsi %32, %28 : index
    %34 = arith.subi %33, %30 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [%c16, %34], strides: [%27, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
    %35 = arith.subi %c64, %34 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [%c16, %35], strides: [%27, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
    %36 = arith.addi %arg5, %c15_i32 : i32
    %37 = arith.divsi %36, %c16_i32 : i32
    %38 = arith.muli %arg7, %c16_i32 : i32
    %39 = arith.index_cast %arg3 : i32 to index
    %40 = arith.index_cast %15 : i32 to index
    %41 = arith.index_cast %arg6 : i32 to index
    %42 = arith.muli %40, %41 : index
    %43 = arith.index_cast %arg7 : i32 to index
    %44 = arith.index_cast %arg4 : i32 to index
    %45 = arith.index_cast %16 : i32 to index
    %46:7 = scf.for %arg15 = %c0_i32 to %37 step %c1_i32 iter_args(%arg16 = %1, %arg17 = %reinterpret_cast, %arg18 = %reinterpret_cast_1, %arg19 = %42, %arg20 = %c0, %arg21 = %reinterpret_cast_0, %arg22 = %reinterpret_cast_2) -> (tensor<32x64xf32>, memref<?x16xf32, strided<[?, ?], offset: ?>>, memref<16x?xf32, strided<[?, ?], offset: ?>>, index, index, memref<?x16xf32, strided<[?, ?], offset: ?>>, memref<16x?xf32, strided<[?, ?], offset: ?>>)  : i32 {
      %64 = arith.muli %arg15, %c16_i32 : i32
      %65 = arith.subi %arg5, %64 : i32
      %alloc = memref.alloc() : memref<32x16xf32>
      %66 = arith.index_cast %65 : i32 to index
      %67 = arith.minsi %66, %c16 : index
      %68 = arith.cmpi slt, %67, %c16 : index
      scf.if %68 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<32x16xf32>)
      }
      %dim = memref.dim %arg17, %c0 : memref<?x16xf32, strided<[?, ?], offset: ?>>
      %69 = arith.minsi %dim, %c32 : index
      %70 = arith.subi %c32, %69 : index
      %subview_4 = memref.subview %arg17[0, 0] [%69, %67] [1, 1] : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_5 = memref.subview %arg21[0, 0] [%70, %67] [1, 1] : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%69, %67] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_7 = memref.subview %alloc[%69, 0] [%70, %67] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      memref.copy %subview_4, %subview_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
      memref.copy %subview_5, %subview_7 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      %71 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32>
      %alloc_8 = memref.alloc() : memref<16x64xf32>
      %72 = arith.index_cast %65 : i32 to index
      %73 = arith.minsi %72, %c16 : index
      %74 = arith.cmpi slt, %73, %c16 : index
      scf.if %74 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<16x64xf32>)
      }
      %dim_9 = memref.dim %arg18, %c1 : memref<16x?xf32, strided<[?, ?], offset: ?>>
      %75 = arith.minsi %dim_9, %c64 : index
      %76 = arith.subi %c64, %75 : index
      %subview_10 = memref.subview %arg18[0, 0] [%73, %75] [1, 1] : memref<16x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %arg22[0, 0] [%73, %76] [1, 1] : memref<16x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_12 = memref.subview %alloc_8[0, 0] [%73, %75] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      %subview_13 = memref.subview %alloc_8[0, %75] [%73, %76] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_10, %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_11, %subview_13 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %77 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x64xf32>
      %78 = tensor.empty() : tensor<32x64xf32>
      %79 = linalg.fill ins(%cst : f32) outs(%78 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %80 = linalg.matmul ins(%71, %77 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%79 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %81 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%80, %arg16 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%80 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_18: f32, %out: f32):
        %98 = arith.addf %in, %in_18 : f32
        linalg.yield %98 : f32
      } -> tensor<32x64xf32>
      %82 = arith.addi %arg19, %c16 : index
      %83 = arith.remsi %82, %41 : index
      %84 = arith.muli %39, %41 : index
      %85 = arith.addi %84, %83 : index
      %86 = arith.subi %85, %82 : index
      %87 = arith.divsi %86, %41 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg0 to offset: [%82], sizes: [%87, %c16], strides: [%41, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
      %88 = arith.subi %c32, %87 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg0 to offset: [%83], sizes: [%88, %c16], strides: [%41, %c1] : memref<*xf32> to memref<?x16xf32, strided<[?, ?], offset: ?>>
      %89 = arith.index_cast %38 : i32 to index
      %90 = arith.addi %arg20, %89 : index
      %91 = arith.addi %90, %45 : index
      %92 = arith.remsi %91, %44 : index
      %93 = arith.subi %91, %92 : index
      %94 = arith.addi %92, %c64 : index
      %95 = arith.minsi %94, %44 : index
      %96 = arith.subi %95, %92 : index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg1 to offset: [%91], sizes: [%c16, %96], strides: [%43, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
      %97 = arith.subi %c64, %96 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg1 to offset: [%93], sizes: [%c16, %97], strides: [%43, %c1] : memref<*xf32> to memref<16x?xf32, strided<[?, ?], offset: ?>>
      scf.yield %81, %reinterpret_cast_14, %reinterpret_cast_16, %82, %90, %reinterpret_cast_15, %reinterpret_cast_17 : tensor<32x64xf32>, memref<?x16xf32, strided<[?, ?], offset: ?>>, memref<16x?xf32, strided<[?, ?], offset: ?>>, index, index, memref<?x16xf32, strided<[?, ?], offset: ?>>, memref<16x?xf32, strided<[?, ?], offset: ?>>
    }
    %47 = arith.index_cast %arg8 : i32 to index
    %48 = arith.index_cast %15 : i32 to index
    %49 = arith.muli %48, %47 : index
    %50 = arith.index_cast %16 : i32 to index
    %51 = arith.addi %49, %50 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%51], sizes: [32, 64], strides: [%47, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
    %52 = arith.index_cast %15 : i32 to index
    %53 = arith.addi %52, %c32 : index
    %54 = arith.index_cast %arg3 : i32 to index
    %55 = arith.minsi %53, %54 : index
    %56 = arith.subi %55, %52 : index
    %57 = arith.index_cast %16 : i32 to index
    %58 = arith.addi %57, %c64 : index
    %59 = arith.index_cast %arg4 : i32 to index
    %60 = arith.minsi %58, %59 : index
    %61 = arith.subi %60, %57 : index
    %62 = arith.minsi %56, %c32 : index
    %63 = arith.minsi %61, %c64 : index
    %extracted_slice = tensor.extract_slice %46#0[0, 0] [%62, %63] [1, 1] : tensor<32x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast_3[0, 0] [%62, %63] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    memref.tensor_store %extracted_slice, %subview : memref<?x?xf32, strided<[?, 1], offset: ?>>
    return
  }
}

