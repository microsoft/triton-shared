#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @_layer_norm_bwd_dwdb_0123456(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: memref<*xf32>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = arith.muli %arg9, %c256_i32 : i32
    %23 = arith.index_cast %2 : i32 to index
    %30 = arith.index_cast %2 : i32 to index
    %45 = arith.index_cast %2 : i32 to index
    %52 = arith.index_cast %2 : i32 to index
    %3:2 = scf.for %arg12 = %c0_i32 to %arg4 step %c256_i32 iter_args(%arg13 = %1, %arg14 = %1) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
      %20 = arith.index_cast %arg12 : i32 to index
      %21 = arith.index_cast %arg5 : i32 to index
      %22 = arith.muli %20, %21 : index
      %24 = arith.addi %22, %23 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%24], sizes: [256, 256], strides: [%21, 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
      %25 = arith.index_cast %arg12 : i32 to index
      %26 = arith.addi %25, %c256 : index
      %27 = arith.index_cast %arg4 : i32 to index
      %28 = arith.minsi %26, %27 : index
      %29 = arith.subi %28, %25 : index
      %31 = arith.addi %30, %c256 : index
      %32 = arith.index_cast %arg5 : i32 to index
      %33 = arith.minsi %31, %32 : index
      %34 = arith.subi %33, %30 : index
      %35 = arith.minsi %29, %c256 : index
      %36 = arith.minsi %34, %c256 : index
      %alloc = memref.alloc() : memref<256x256xf32>
      %37 = arith.cmpi slt, %35, %c256 : index
      %38 = arith.cmpi slt, %36, %c256 : index
      %39 = arith.ori %37, %38 : i1
      scf.if %39 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<256x256xf32>)
      }
      %subview_5 = memref.subview %reinterpret_cast_4[0, 0] [%35, %36] [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%35, %36] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
      %40 = bufferization.to_tensor %alloc restrict writable : memref<256x256xf32>
      %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg13, %40 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%arg13 : tensor<256x256xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %64 = arith.addf %in, %in_11 : f32
        linalg.yield %64 : f32
      } -> tensor<256x256xf32>
      %42 = arith.index_cast %arg12 : i32 to index
      %43 = arith.index_cast %arg5 : i32 to index
      %44 = arith.muli %42, %43 : index
      %46 = arith.addi %44, %45 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%46], sizes: [256, 256], strides: [%43, 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<256x256xf32>
      %47 = arith.index_cast %arg12 : i32 to index
      %48 = arith.addi %47, %c256 : index
      %49 = arith.index_cast %arg4 : i32 to index
      %50 = arith.minsi %48, %49 : index
      %51 = arith.subi %50, %47 : index
      %53 = arith.addi %52, %c256 : index
      %54 = arith.index_cast %arg5 : i32 to index
      %55 = arith.minsi %53, %54 : index
      %56 = arith.subi %55, %52 : index
      %57 = arith.minsi %51, %c256 : index
      %58 = arith.minsi %56, %c256 : index
      %59 = arith.cmpi slt, %57, %c256 : index
      %60 = arith.cmpi slt, %58, %c256 : index
      %61 = arith.ori %59, %60 : i1
      scf.if %61 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<256x256xf32>)
      }
      %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%57, %58] [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0, 0] [%57, %58] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
      memref.copy %subview_9, %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
      %62 = bufferization.to_tensor %alloc_8 restrict writable : memref<256x256xf32>
      %63 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %62 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%arg14 : tensor<256x256xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %64 = arith.addf %in, %in_11 : f32
        linalg.yield %64 : f32
      } -> tensor<256x256xf32>
      scf.yield %41, %63 : tensor<256x256xf32>, tensor<256x256xf32>
    }
    %4 = tensor.empty() : tensor<256xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%3#0 : tensor<256x256xf32>) outs(%5 : tensor<256xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %6 = tensor.empty() : tensor<256xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<256xf32>) -> tensor<256xf32>
    %reduced_0 = linalg.reduce ins(%3#1 : tensor<256x256xf32>) outs(%7 : tensor<256xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %8 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %9 = arith.index_cast %2 : i32 to index
    %10 = arith.addi %9, %c256 : index
    %11 = arith.index_cast %arg5 : i32 to index
    %12 = arith.minsi %10, %11 : index
    %13 = arith.subi %12, %9 : index
    %extracted_slice = tensor.extract_slice %reduced[0] [%13] [1] : tensor<256xf32> to tensor<?xf32>
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %14 = arith.index_cast %2 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %15 = arith.index_cast %2 : i32 to index
    %16 = arith.addi %15, %c256 : index
    %17 = arith.index_cast %arg5 : i32 to index
    %18 = arith.minsi %16, %17 : index
    %19 = arith.subi %18, %15 : index
    %extracted_slice_2 = tensor.extract_slice %reduced_0[0] [%19] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%19] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_2 in writable %subview_3 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

