#map = affine_map<(d0) -> (d0)>
module {
  func.func @softmax_kernel_012345(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c128 = arith.constant 128 : index
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<128xf32>
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.minsi %2, %c128 : index
    %4 = arith.cmpi slt, %3, %c128 : index
    scf.if %4 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<128xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%3] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<128xf32>
    %6 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst_0 into %6[] : tensor<f32>
    %reduced = linalg.reduce ins(%5 : tensor<128xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %19 = arith.maximumf %in, %init : f32
        linalg.yield %19 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %7 = tensor.empty() : tensor<128xf32>
    %8 = linalg.fill ins(%extracted : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %8 : tensor<128xf32>, tensor<128xf32>) outs(%5 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %19 = arith.subf %in, %in_7 : f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%9 : tensor<128xf32>) outs(%9 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = math.exp %in : f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    %11 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst into %11[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%10 : tensor<128xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %19 = arith.addf %in, %init : f32
        linalg.yield %19 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %12 = tensor.empty() : tensor<128xf32>
    %13 = linalg.fill ins(%extracted_4 : f32) outs(%12 : tensor<128xf32>) -> tensor<128xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %13 : tensor<128xf32>, tensor<128xf32>) outs(%10 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %19 = arith.divf %in, %in_7 : f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    %15 = arith.muli %arg8, %arg3 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %17 = arith.index_cast %arg4 : i32 to index
    %18 = arith.minsi %17, %c128 : index
    %extracted_slice = tensor.extract_slice %14[0] [%18] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%18] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

