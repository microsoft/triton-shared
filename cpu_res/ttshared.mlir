#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel_0d1d2d3de(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %4 = arith.index_cast %0 : i32 to index
    %5 = arith.addi %4, %c1024 : index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.minsi %5, %6 : index
    %8 = arith.subi %7, %4 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %subview = memref.subview %reinterpret_cast[0] [%8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%8] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %10 = arith.index_cast %0 : i32 to index
    %11 = arith.addi %10, %c1024 : index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.minsi %11, %12 : index
    %14 = arith.subi %13, %10 : index
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%14] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%14] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %15 : tensor<1024xf32>, tensor<1024xf32>) outs(%9 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %22 = arith.addf %in, %in_6 : f32
      linalg.yield %22 : f32
    } -> tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %17 = arith.index_cast %0 : i32 to index
    %18 = arith.addi %17, %c1024 : index
    %19 = arith.index_cast %arg3 : i32 to index
    %20 = arith.minsi %18, %19 : index
    %21 = arith.subi %20, %17 : index
    return
  }
}

