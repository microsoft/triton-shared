module {
  func.func @triton__0d1d2de(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c4_i32 = arith.constant 4 : i32
    %c128 = arith.constant 128 : index
    %c77_i32 = arith.constant 77 : i32
    %c4 = arith.constant 4 : index
    %0 = arith.muli %arg6, %c4_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %1, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4, 64], strides: [%c128, 1] : memref<*xi32> to memref<4x64xi32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x64xi32>
    linalg.fill ins(%c77_i32 : i32) outs(%alloc : memref<4x64xi32>)
    %subview = memref.subview %reinterpret_cast[0, 0] [4, 32] [1, 1] : memref<4x64xi32, strided<[?, 1], offset: ?>> to memref<4x32xi32, strided<[?, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0] [4, 32] [1, 1] : memref<4x64xi32> to memref<4x32xi32, strided<[64, 1]>>
    memref.copy %subview, %subview_0 : memref<4x32xi32, strided<[?, 1], offset: ?>> to memref<4x32xi32, strided<[64, 1]>>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<4x64xi32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 64], strides: [1, %c4] : memref<*xi32> to memref<4x64xi32, strided<[1, ?]>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_1 : (tensor<4x64xi32>, memref<4x64xi32, strided<[1, ?]>>) -> ()
    return
  }
}

