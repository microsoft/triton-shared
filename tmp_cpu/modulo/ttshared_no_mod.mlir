module {
  func.func @mod_2d_0d1d2e3e4e5c67c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %0 = arith.index_cast %arg4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 4], strides: [%0, 1] : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
    %1 = arith.index_cast %arg5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 4], strides: [%1, 1] : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<4x4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x4xf32, strided<[?, 1]>> to memref<4x4xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
    memref.tensor_store %2, %reinterpret_cast_0 : memref<4x4xf32, strided<[?, 1]>>
    return
  }
}

