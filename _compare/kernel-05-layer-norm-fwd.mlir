#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused_0123456789(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: memref<*xf32>, %arg4: memref<*xf32>, %arg5: memref<*xf32>, %arg6: i32, %arg7: i32, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %2 = arith.muli %arg12, %arg6 : i32
    %3 = scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg16 = %1) -> (tensor<256xf32>)  : i32 {
      %25 = arith.index_cast %2 : i32 to index
      %26 = arith.index_cast %arg15 : i32 to index
      %27 = arith.addi %25, %26 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<256xf32>
      %28 = arith.index_cast %arg15 : i32 to index
      %29 = arith.addi %28, %c256 : index
      %30 = arith.index_cast %arg7 : i32 to index
      %31 = arith.minsi %29, %30 : index
      %32 = arith.subi %31, %28 : index
      %33 = arith.cmpi slt, %32, %c256 : index
      scf.if %33 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
      }
      %subview = memref.subview %reinterpret_cast_5[0] [%32] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %alloc[0] [%32] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %34 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %34 : tensor<256xf32>, tensor<256xf32>) outs(%arg16 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %36 = arith.addf %in, %in_7 : f32
        linalg.yield %36 : f32
      } -> tensor<256xf32>
      scf.yield %35 : tensor<256xf32>
    }
    %4 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst_0 into %4[] : tensor<f32>
    %reduced = linalg.reduce ins(%3 : tensor<256xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %25 = arith.addf %in, %init : f32
        linalg.yield %25 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %5 = arith.sitofp %arg7 : i32 to f32
    %6 = arith.divf %extracted, %5 : f32
    %7 = tensor.empty() : tensor<256xi32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%7 : tensor<256xi32>) {
    ^bb0(%out: i32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i32
      linalg.yield %26 : i32
    } -> tensor<256xi32>
    %9 = tensor.empty() : tensor<256xi32>
    %10 = linalg.fill ins(%arg7 : i32) outs(%9 : tensor<256xi32>) -> tensor<256xi32>
    %11 = tensor.empty() : tensor<256xf32>
    %12 = linalg.fill ins(%6 : f32) outs(%11 : tensor<256xf32>) -> tensor<256xf32>
    %13 = scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg16 = %1) -> (tensor<256xf32>)  : i32 {
      %25 = tensor.empty() : tensor<256xi32>
      %26 = linalg.fill ins(%arg15 : i32) outs(%25 : tensor<256xi32>) -> tensor<256xi32>
      %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%26, %8 : tensor<256xi32>, tensor<256xi32>) outs(%26 : tensor<256xi32>) {
      ^bb0(%in: i32, %in_7: i32, %out: i32):
        %44 = arith.addi %in, %in_7 : i32
        linalg.yield %44 : i32
      } -> tensor<256xi32>
      %28 = tensor.empty() : tensor<256xi1>
      %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%27, %10 : tensor<256xi32>, tensor<256xi32>) outs(%28 : tensor<256xi1>) {
      ^bb0(%in: i32, %in_7: i32, %out: i1):
        %44 = arith.cmpi slt, %in, %in_7 : i32
        linalg.yield %44 : i1
      } -> tensor<256xi1>
      %30 = arith.index_cast %2 : i32 to index
      %31 = arith.index_cast %arg15 : i32 to index
      %32 = arith.addi %30, %31 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%32], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<256xf32>
      %33 = arith.index_cast %arg15 : i32 to index
      %34 = arith.addi %33, %c256 : index
      %35 = arith.index_cast %arg7 : i32 to index
      %36 = arith.minsi %34, %35 : index
      %37 = arith.subi %36, %33 : index
      %38 = arith.cmpi slt, %37, %c256 : index
      scf.if %38 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
      }
      %subview = memref.subview %reinterpret_cast_5[0] [%37] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %alloc[0] [%37] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %39 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%39, %12 : tensor<256xf32>, tensor<256xf32>) outs(%39 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.subf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<256xf32>
      %41 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%29, %40, %1 : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%40 : tensor<256xf32>) {
      ^bb0(%in: i1, %in_7: f32, %in_8: f32, %out: f32):
        %44 = arith.select %in, %in_7, %in_8 : f32
        linalg.yield %44 : f32
      } -> tensor<256xf32>
      %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%41, %41 : tensor<256xf32>, tensor<256xf32>) outs(%41 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.mulf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<256xf32>
      %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %42 : tensor<256xf32>, tensor<256xf32>) outs(%arg16 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.addf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<256xf32>
      scf.yield %43 : tensor<256xf32>
    }
    %14 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_1 = tensor.insert %cst_0 into %14[] : tensor<f32>
    %reduced_2 = linalg.reduce ins(%13 : tensor<256xf32>) outs(%inserted_1 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %25 = arith.addf %in, %init : f32
        linalg.yield %25 : f32
      }
    %extracted_3 = tensor.extract %reduced_2[] : tensor<f32>
    %15 = arith.divf %extracted_3, %5 : f32
    %16 = arith.addf %15, %arg8 : f32
    %17 = math.sqrt %16 : f32
    %18 = arith.divf %cst, %17 : f32
    %19 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %6, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %20 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%20], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %18, %reinterpret_cast_4[0] : memref<1xf32, strided<[1], offset: ?>>
    %21 = tensor.empty() : tensor<256xf32>
    %22 = linalg.fill ins(%6 : f32) outs(%21 : tensor<256xf32>) -> tensor<256xf32>
    %23 = tensor.empty() : tensor<256xf32>
    %24 = linalg.fill ins(%18 : f32) outs(%23 : tensor<256xf32>) -> tensor<256xf32>
    scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %25 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<256xf32>
      %26 = arith.index_cast %arg15 : i32 to index
      %27 = arith.addi %26, %c256 : index
      %28 = arith.index_cast %arg7 : i32 to index
      %29 = arith.minsi %27, %28 : index
      %30 = arith.subi %29, %26 : index
      %subview = memref.subview %reinterpret_cast_5[0] [%30] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %alloc[0] [%30] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %31 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %32 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%32], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<256xf32>
      %33 = arith.index_cast %arg15 : i32 to index
      %34 = arith.addi %33, %c256 : index
      %35 = arith.index_cast %arg7 : i32 to index
      %36 = arith.minsi %34, %35 : index
      %37 = arith.subi %36, %33 : index
      %subview_9 = memref.subview %reinterpret_cast_7[0] [%37] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0] [%37] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_9, %subview_10 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %38 = bufferization.to_tensor %alloc_8 restrict writable : memref<256xf32>
      %39 = arith.index_cast %2 : i32 to index
      %40 = arith.index_cast %arg15 : i32 to index
      %41 = arith.addi %39, %40 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<256xf32>
      %42 = arith.index_cast %arg15 : i32 to index
      %43 = arith.addi %42, %c256 : index
      %44 = arith.index_cast %arg7 : i32 to index
      %45 = arith.minsi %43, %44 : index
      %46 = arith.subi %45, %42 : index
      %47 = arith.cmpi slt, %46, %c256 : index
      scf.if %47 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_12 : memref<256xf32>)
      }
      %subview_13 = memref.subview %reinterpret_cast_11[0] [%46] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_14 = memref.subview %alloc_12[0] [%46] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_13, %subview_14 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %48 = bufferization.to_tensor %alloc_12 restrict writable : memref<256xf32>
      %49 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%48, %22 : tensor<256xf32>, tensor<256xf32>) outs(%48 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %61 = arith.subf %in, %in_17 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%49, %24 : tensor<256xf32>, tensor<256xf32>) outs(%49 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %61 = arith.mulf %in, %in_17 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %51 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%50, %31 : tensor<256xf32>, tensor<256xf32>) outs(%50 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %61 = arith.mulf %in, %in_17 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %52 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%51, %38 : tensor<256xf32>, tensor<256xf32>) outs(%51 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %61 = arith.addf %in, %in_17 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %53 = arith.index_cast %2 : i32 to index
      %54 = arith.index_cast %arg15 : i32 to index
      %55 = arith.addi %53, %54 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg1 to offset: [%55], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %56 = arith.index_cast %arg15 : i32 to index
      %57 = arith.addi %56, %c256 : index
      %58 = arith.index_cast %arg7 : i32 to index
      %59 = arith.minsi %57, %58 : index
      %60 = arith.subi %59, %56 : index
      %extracted_slice = tensor.extract_slice %52[0] [%60] [1] : tensor<256xf32> to tensor<?xf32>
      %subview_16 = memref.subview %reinterpret_cast_15[0] [%60] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_16 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

