#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused_0123456789(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: memref<*xf32>, %arg4: memref<*xf32>, %arg5: memref<*xf32>, %arg6: i32, %arg7: i32, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c256 = arith.constant 256 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %2 = arith.muli %arg12, %arg6 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %5 = scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg16 = %1) -> (tensor<256xf32>)  : i32 {
      %27 = arith.index_cast %arg15 : i32 to index
      %28 = arith.addi %3, %27 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %29 = arith.index_cast %arg15 : i32 to index
      %30 = arith.addi %29, %c256 : index
      %31 = arith.index_cast %arg7 : i32 to index
      %32 = arith.minsi %30, %31 : index
      %33 = arith.subi %32, %29 : index
      %alloc = memref.alloc() : memref<256xf32>
      %34 = arith.cmpi slt, %33, %c256 : index
      scf.if %34 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
      }
      %subview = memref.subview %reinterpret_cast_11[0] [%33] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %alloc[0] [%33] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_12 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %35 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %35 : tensor<256xf32>, tensor<256xf32>) outs(%arg16 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %37 = arith.addf %in, %in_13 : f32
        linalg.yield %37 : f32
      } -> tensor<256xf32>
      scf.yield %36 : tensor<256xf32>
    }
    %6 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst_0 into %6[] : tensor<f32>
    %reduced = linalg.reduce ins(%5 : tensor<256xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %27 = arith.addf %in, %init : f32
        linalg.yield %27 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %7 = arith.sitofp %arg7 : i32 to f32
    %8 = arith.divf %extracted, %7 : f32
    %9 = tensor.empty() : tensor<256xi32>
    %10 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%9 : tensor<256xi32>) {
    ^bb0(%out: i32):
      %27 = linalg.index 0 : index
      %28 = arith.index_cast %27 : index to i32
      linalg.yield %28 : i32
    } -> tensor<256xi32>
    %11 = tensor.empty() : tensor<256xi32>
    %12 = linalg.fill ins(%arg7 : i32) outs(%11 : tensor<256xi32>) -> tensor<256xi32>
    %13 = tensor.empty() : tensor<256xf32>
    %14 = linalg.fill ins(%8 : f32) outs(%13 : tensor<256xf32>) -> tensor<256xf32>
    %15 = scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg16 = %1) -> (tensor<256xf32>)  : i32 {
      %27 = tensor.empty() : tensor<256xi32>
      %28 = linalg.fill ins(%arg15 : i32) outs(%27 : tensor<256xi32>) -> tensor<256xi32>
      %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%28, %10 : tensor<256xi32>, tensor<256xi32>) outs(%28 : tensor<256xi32>) {
      ^bb0(%in: i32, %in_13: i32, %out: i32):
        %45 = arith.addi %in, %in_13 : i32
        linalg.yield %45 : i32
      } -> tensor<256xi32>
      %30 = tensor.empty() : tensor<256xi1>
      %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%29, %12 : tensor<256xi32>, tensor<256xi32>) outs(%30 : tensor<256xi1>) {
      ^bb0(%in: i32, %in_13: i32, %out: i1):
        %45 = arith.cmpi slt, %in, %in_13 : i32
        linalg.yield %45 : i1
      } -> tensor<256xi1>
      %32 = arith.index_cast %arg15 : i32 to index
      %33 = arith.addi %3, %32 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg0 to offset: [%33], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %34 = arith.index_cast %arg15 : i32 to index
      %35 = arith.addi %34, %c256 : index
      %36 = arith.index_cast %arg7 : i32 to index
      %37 = arith.minsi %35, %36 : index
      %38 = arith.subi %37, %34 : index
      %alloc = memref.alloc() : memref<256xf32>
      %39 = arith.cmpi slt, %38, %c256 : index
      scf.if %39 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
      }
      %subview = memref.subview %reinterpret_cast_11[0] [%38] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %alloc[0] [%38] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_12 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %40 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%40, %14 : tensor<256xf32>, tensor<256xf32>) outs(%40 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %45 = arith.subf %in, %in_13 : f32
        linalg.yield %45 : f32
      } -> tensor<256xf32>
      %42 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%31, %41, %1 : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%41 : tensor<256xf32>) {
      ^bb0(%in: i1, %in_13: f32, %in_14: f32, %out: f32):
        %45 = arith.select %in, %in_13, %in_14 : f32
        linalg.yield %45 : f32
      } -> tensor<256xf32>
      %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%42, %42 : tensor<256xf32>, tensor<256xf32>) outs(%42 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %45 = arith.mulf %in, %in_13 : f32
        linalg.yield %45 : f32
      } -> tensor<256xf32>
      %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %43 : tensor<256xf32>, tensor<256xf32>) outs(%arg16 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %45 = arith.addf %in, %in_13 : f32
        linalg.yield %45 : f32
      } -> tensor<256xf32>
      scf.yield %44 : tensor<256xf32>
    }
    %16 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst_0 into %16[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%15 : tensor<256xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %27 = arith.addf %in, %init : f32
        linalg.yield %27 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %17 = arith.divf %extracted_4, %7 : f32
    %18 = arith.addf %17, %arg8 : f32
    %19 = math.sqrt %18 : f32
    %20 = arith.divf %cst, %19 : f32
    %21 = arith.index_cast %arg12 : i32 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast_1 : memref<1xf32> -> memref<f32>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [%21], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %8, %reinterpret_cast_5[0] : memref<1xf32, strided<[1], offset: ?>>
    %22 = arith.index_cast %arg12 : i32 to index
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %reinterpret_cast : memref<1xf32> -> memref<f32>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [%22], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %20, %reinterpret_cast_10[0] : memref<1xf32, strided<[1], offset: ?>>
    %23 = tensor.empty() : tensor<256xf32>
    %24 = linalg.fill ins(%8 : f32) outs(%23 : tensor<256xf32>) -> tensor<256xf32>
    %25 = tensor.empty() : tensor<256xf32>
    %26 = linalg.fill ins(%20 : f32) outs(%25 : tensor<256xf32>) -> tensor<256xf32>
    scf.for %arg15 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %27 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %28 = arith.index_cast %arg15 : i32 to index
      %29 = arith.addi %28, %c256 : index
      %30 = arith.index_cast %arg7 : i32 to index
      %31 = arith.minsi %29, %30 : index
      %32 = arith.subi %31, %28 : index
      %alloc = memref.alloc() : memref<256xf32>
      %subview = memref.subview %reinterpret_cast_11[0] [%32] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %alloc[0] [%32] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_12 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %33 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
      %34 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %35 = arith.index_cast %arg15 : i32 to index
      %36 = arith.addi %35, %c256 : index
      %37 = arith.index_cast %arg7 : i32 to index
      %38 = arith.minsi %36, %37 : index
      %39 = arith.subi %38, %35 : index
      %alloc_14 = memref.alloc() : memref<256xf32>
      %subview_15 = memref.subview %reinterpret_cast_13[0] [%39] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_16 = memref.subview %alloc_14[0] [%39] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_15, %subview_16 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %40 = bufferization.to_tensor %alloc_14 restrict writable : memref<256xf32>
      %41 = arith.index_cast %arg15 : i32 to index
      %42 = arith.addi %3, %41 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg0 to offset: [%42], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %43 = arith.index_cast %arg15 : i32 to index
      %44 = arith.addi %43, %c256 : index
      %45 = arith.index_cast %arg7 : i32 to index
      %46 = arith.minsi %44, %45 : index
      %47 = arith.subi %46, %43 : index
      %alloc_18 = memref.alloc() : memref<256xf32>
      %48 = arith.cmpi slt, %47, %c256 : index
      scf.if %48 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_18 : memref<256xf32>)
      }
      %subview_19 = memref.subview %reinterpret_cast_17[0] [%47] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_20 = memref.subview %alloc_18[0] [%47] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_19, %subview_20 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %49 = bufferization.to_tensor %alloc_18 restrict writable : memref<256xf32>
      %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%49, %24 : tensor<256xf32>, tensor<256xf32>) outs(%49 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_23: f32, %out: f32):
        %61 = arith.subf %in, %in_23 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %51 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%50, %26 : tensor<256xf32>, tensor<256xf32>) outs(%50 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_23: f32, %out: f32):
        %61 = arith.mulf %in, %in_23 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %52 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%51, %33 : tensor<256xf32>, tensor<256xf32>) outs(%51 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_23: f32, %out: f32):
        %61 = arith.mulf %in, %in_23 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %53 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%52, %40 : tensor<256xf32>, tensor<256xf32>) outs(%52 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_23: f32, %out: f32):
        %61 = arith.addf %in, %in_23 : f32
        linalg.yield %61 : f32
      } -> tensor<256xf32>
      %54 = arith.index_cast %arg15 : i32 to index
      %55 = arith.addi %4, %54 : index
      %reinterpret_cast_21 = memref.reinterpret_cast %arg1 to offset: [%55], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
      %56 = arith.index_cast %arg15 : i32 to index
      %57 = arith.addi %56, %c256 : index
      %58 = arith.index_cast %arg7 : i32 to index
      %59 = arith.minsi %57, %58 : index
      %60 = arith.subi %59, %56 : index
      %extracted_slice = tensor.extract_slice %53[0] [%60] [1] : tensor<256xf32> to tensor<?xf32>
      %subview_22 = memref.subview %reinterpret_cast_21[0] [%60] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_22 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

