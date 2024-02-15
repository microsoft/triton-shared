#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_argmax(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i32
      linalg.yield %14 : i32
    } -> tensor<4xi32>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 4], strides: [%2, %3] : memref<*xf32> to memref<4x4xf32, strided<[?, ?]>>
    %alloc = memref.alloc() : memref<4x4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x4xf32, strided<[?, ?]>> to memref<4x4xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
    %5 = tensor.empty() : tensor<4x4xi32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<1x4xi32>) outs(%5 : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<4x4xi32>
    %7 = tensor.empty() : tensor<4xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
    %9 = tensor.empty() : tensor<4xi32>
    %10 = linalg.fill ins(%c-1_i32 : i32) outs(%9 : tensor<4xi32>) -> tensor<4xi32>
    %reduced:2 = linalg.reduce ins(%4, %6 : tensor<4x4xf32>, tensor<4x4xi32>) outs(%8, %10 : tensor<4xf32>, tensor<4xi32>) dimensions = [1] 
      (%in: f32, %in_1: i32, %init: f32, %init_2: i32) {
        %13 = arith.cmpf oeq, %in, %init : f32
        %14 = arith.cmpi slt, %in_1, %init_2 : i32
        %15 = arith.andi %13, %14 : i1
        %16 = arith.cmpf ogt, %in, %init : f32
        %17 = arith.ori %16, %15 : i1
        %18 = arith.select %17, %in, %init : f32
        %19 = arith.select %17, %in_1, %init_2 : i32
        linalg.yield %18, %19 : f32, i32
      }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %11 = tensor.empty() : tensor<4xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%reduced#1 : tensor<4xi32>) outs(%11 : tensor<4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %13 = arith.sitofp %in : i32 to f32
      linalg.yield %13 : f32
    } -> tensor<4xf32>
    bufferization.materialize_in_destination %12 in writable %reinterpret_cast_0 : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_argmin(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0x7F800000 : f32
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i32
      linalg.yield %14 : i32
    } -> tensor<4xi32>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 4], strides: [%2, %3] : memref<*xf32> to memref<4x4xf32, strided<[?, ?]>>
    %alloc = memref.alloc() : memref<4x4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x4xf32, strided<[?, ?]>> to memref<4x4xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
    %5 = tensor.empty() : tensor<4x4xi32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<1x4xi32>) outs(%5 : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<4x4xi32>
    %7 = tensor.empty() : tensor<4xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
    %9 = tensor.empty() : tensor<4xi32>
    %10 = linalg.fill ins(%c-1_i32 : i32) outs(%9 : tensor<4xi32>) -> tensor<4xi32>
    %reduced:2 = linalg.reduce ins(%4, %6 : tensor<4x4xf32>, tensor<4x4xi32>) outs(%8, %10 : tensor<4xf32>, tensor<4xi32>) dimensions = [1] 
      (%in: f32, %in_1: i32, %init: f32, %init_2: i32) {
        %13 = arith.cmpf oeq, %in, %init : f32
        %14 = arith.cmpi slt, %in_1, %init_2 : i32
        %15 = arith.andi %13, %14 : i1
        %16 = arith.cmpf olt, %in, %init : f32
        %17 = arith.ori %16, %15 : i1
        %18 = arith.select %17, %in, %init : f32
        %19 = arith.select %17, %in_1, %init_2 : i32
        linalg.yield %18, %19 : f32, i32
      }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %11 = tensor.empty() : tensor<4xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%reduced#1 : tensor<4xi32>) outs(%11 : tensor<4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %13 = arith.sitofp %in : i32 to f32
      linalg.yield %13 : f32
    } -> tensor<4xf32>
    bufferization.materialize_in_destination %12 in writable %reinterpret_cast_0 : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
    return
  }
}

