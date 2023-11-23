module {
  func.func @mod_2d_0d1d2e3e4e5c67c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.remsi %c0, %1 : index // %2 = 0 % (%1) = 0
    %3 = arith.subi %c0, %2 : index // %3 = 0 - %2 = 0
    %4 = arith.addi %2, %c4 : index // %4 = 0 + c4 = 4
    %5 = arith.minsi %4, %1 : index // %5 = min(%4, %1) = 4
    %6 = arith.subi %5, %2 : index // %6 = 4 - 4 = 0
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [%c4, %6], strides: [%0, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %7 = arith.subi %c4, %6 : index // %7 = 4 - %6 = 4
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [%c4, %7], strides: [%0, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %8 = arith.index_cast %arg5 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 4], strides: [%8, 1] : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<4x4xf32>
    %subview = memref.subview %alloc[0, 0] [4, %6] [%0, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[?, 1]>>
    %subview_2 = memref.subview %alloc[0, %6] [4, %7] [%0, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[?, 1], offset: ?>>
    memref.copy %reinterpret_cast, %subview : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, 1]>>
    memref.copy %reinterpret_cast_0, %subview_2 : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, 1], offset: ?>>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
    memref.tensor_store %9, %reinterpret_cast_1 : memref<4x4xf32, strided<[?, 1]>>
    return
  }
}

