module {
  func.func @mod_2d_0d1d2e3e4e5c67c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.remsi %c0, %1 : index
    %3 = arith.addi %2, %c4 : index
    %4 = arith.minsi %3, %1 : index
    %5 = arith.subi %4, %2 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [%c4, %5], strides: [%0, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %6 = arith.index_cast %arg5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 4], strides: [%6, 1] : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<4x4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x4xf32>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
    memref.tensor_store %7, %reinterpret_cast_0 : memref<4x4xf32, strided<[?, 1]>>
    return
  }
}

