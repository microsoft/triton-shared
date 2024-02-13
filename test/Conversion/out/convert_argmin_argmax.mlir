#map = affine_map<(d0) -> (d0)>
module {
  func.func @argmax_012(%arg0: memref<*xf32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0xFF800000 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32>
    %0 = arith.muli %arg6, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = tensor.empty() : tensor<4096xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<4096xi32>) {
    ^bb0(%out: i32):
      %10 = linalg.index 0 : index
      %11 = arith.index_cast %10 : index to i32
      linalg.yield %11 : i32
    } -> tensor<4096xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<4096xf32>
    memref.copy %reinterpret_cast_0, %alloc : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4096xf32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
    %7 = tensor.empty() : tensor<i32>
    %8 = linalg.fill ins(%c-1_i32 : i32) outs(%7 : tensor<i32>) -> tensor<i32>
    %reduced:2 = linalg.reduce ins(%4, %3 : tensor<4096xf32>, tensor<4096xi32>) outs(%6, %8 : tensor<f32>, tensor<i32>) dimensions = [0] 
      (%in: f32, %in_2: i32, %init: f32, %init_3: i32) {
        %10 = arith.cmpf oeq, %in, %init : f32
        %11 = arith.cmpi slt, %in_2, %init_3 : i32
        %12 = arith.andi %10, %11 : i1
        %13 = arith.cmpf ogt, %in, %init : f32
        %14 = arith.ori %13, %12 : i1
        %15 = arith.select %14, %in, %init : f32
        %16 = arith.select %14, %in_2, %init_3 : i32
        linalg.yield %15, %16 : f32, i32
      }
    %extracted = tensor.extract %reduced#1[] : tensor<i32>
    %9 = arith.index_cast %arg6 : i32 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast : memref<1xi32> -> memref<i32>, index, index, index
    %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer to offset: [%9], sizes: [1], strides: [1] : memref<i32> to memref<1xi32, strided<[1], offset: ?>>
    affine.store %extracted, %reinterpret_cast_1[0] : memref<1xi32, strided<[1], offset: ?>>
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @argmin_012(%arg0: memref<*xf32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0x7F800000 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32>
    %0 = arith.muli %arg6, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = tensor.empty() : tensor<4096xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<4096xi32>) {
    ^bb0(%out: i32):
      %10 = linalg.index 0 : index
      %11 = arith.index_cast %10 : index to i32
      linalg.yield %11 : i32
    } -> tensor<4096xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<4096xf32>
    memref.copy %reinterpret_cast_0, %alloc : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4096xf32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
    %7 = tensor.empty() : tensor<i32>
    %8 = linalg.fill ins(%c-1_i32 : i32) outs(%7 : tensor<i32>) -> tensor<i32>
    %reduced:2 = linalg.reduce ins(%4, %3 : tensor<4096xf32>, tensor<4096xi32>) outs(%6, %8 : tensor<f32>, tensor<i32>) dimensions = [0] 
      (%in: f32, %in_2: i32, %init: f32, %init_3: i32) {
        %10 = arith.cmpf oeq, %in, %init : f32
        %11 = arith.cmpi slt, %in_2, %init_3 : i32
        %12 = arith.andi %10, %11 : i1
        %13 = arith.cmpf olt, %in, %init : f32
        %14 = arith.ori %13, %12 : i1
        %15 = arith.select %14, %in, %init : f32
        %16 = arith.select %14, %in_2, %init_3 : i32
        linalg.yield %15, %16 : f32, i32
      }
    %extracted = tensor.extract %reduced#1[] : tensor<i32>
    %9 = arith.index_cast %arg6 : i32 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast : memref<1xi32> -> memref<i32>, index, index, index
    %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer to offset: [%9], sizes: [1], strides: [1] : memref<i32> to memref<1xi32, strided<[1], offset: ?>>
    affine.store %extracted, %reinterpret_cast_1[0] : memref<1xi32, strided<[1], offset: ?>>
    return
  }
}

