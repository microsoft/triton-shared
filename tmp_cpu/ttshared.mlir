module {
  func.func @wrap_side_by_side_loop_0d1d23e4e5c6e7c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.max_divisibility = 8 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.remsi %c6, %1 : index
    %3 = arith.subi %c6, %2 : index
    %4 = arith.addi %2, %c4 : index
    %5 = arith.minsi %4, %1 : index
    %6 = arith.subi %5, %2 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c6], sizes: [%c4, %6], strides: [%0, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %7 = arith.subi %c4, %6 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [%c4, %7], strides: [%0, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = arith.muli %arg4, %c4_i32 : i32
    %10 = arith.index_cast %arg4 : i32 to index
    %11 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [4, 4], strides: [%8, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %12:6 = scf.for %arg12 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg13 = %reinterpret_cast, %arg14 = %reinterpret_cast_1, %arg15 = %c0, %arg16 = %c0, %arg17 = %c0, %arg18 = %reinterpret_cast_0) -> (memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>)  : i32 {
      // %reinterpret_cast_2 = memref.reinterpret_cast %arg13 to offset: [%c0], sizes: [%c4, %6], strides: [%0, %c1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      // %reinterpret_cast_3 = memref.reinterpret_cast %arg18 to offset: [%c0], sizes: [%c4, %7], strides: [%0, %c1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<4x4xf32>
      %subview = memref.subview %alloc[0, 0] [4, %6] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[0, %6] [4, %7] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1], offset: ?>>
      memref.copy %arg13, %subview : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[4, 1]>>
      memref.copy %arg18, %subview_4 : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[4, 1], offset: ?>>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      memref.tensor_store %13, %arg14 : memref<4x4xf32, strided<[?, ?], offset: ?>>
      %14 = arith.index_cast %9 : i32 to index
      %15 = arith.addi %arg15, %14 : index
      %16 = arith.addi %15, %c6 : index
      %17 = arith.remsi %16, %11 : index
      %18 = arith.subi %16, %17 : index
      %19 = arith.addi %17, %c4 : index
      %20 = arith.minsi %19, %11 : index
      %21 = arith.subi %20, %17 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [%c4, %21], strides: [%10, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %22 = arith.subi %c4, %21 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [%c4, %22], strides: [%10, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %23 = arith.index_cast %9 : i32 to index
      %24 = arith.addi %arg16, %23 : index
      %25 = arith.addi %24, %arg17 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [4, 4], strides: [%8, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      scf.yield %reinterpret_cast_5, %reinterpret_cast_7, %15, %25, %c0, %reinterpret_cast_6 : memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>
    }
    return
  }
}

