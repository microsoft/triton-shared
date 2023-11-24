module {
  func.func @wrap_side_by_side_masked_loop_0d1d2e3e4e5c6e7c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}, %arg3: i32 {tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.max_divisibility = 8 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.muli %0, %c2 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.addi %1, %c6 : index
    %4 = arith.remsi %3, %2 : index
    %5 = arith.subi %3, %4 : index
    %6 = arith.addi %4, %c4 : index
    %7 = arith.minsi %6, %2 : index
    %8 = arith.subi %7, %4 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [%c4, %8], strides: [%8, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %9 = arith.subi %c4, %8 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [%c4, %9], strides: [%9, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %10 = arith.index_cast %arg5 : i32 to index
    %11 = arith.muli %arg4, %c4_i32 : i32
    %12 = arith.index_cast %arg4 : i32 to index
    %13 = arith.muli %12, %c2 : index
    %14 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [4, 4], strides: [%10, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %15:6 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %reinterpret_cast, %arg14 = %reinterpret_cast_1, %arg15 = %13, %arg16 = %c0, %arg17 = %c0, %arg18 = %reinterpret_cast_0) -> (memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>)  : i32 {
      %reinterpret_cast_2 = memref.reinterpret_cast %arg13 to offset: [%c0], sizes: [%c4, %8], strides: [%8, %c1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg18 to offset: [%c0], sizes: [%c4, %9], strides: [%9, %c1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %16 = arith.minsi %8, %c4 : index
      %17 = arith.subi %c4, %16 : index
      %subview = memref.subview %reinterpret_cast_2[0, 0] [2, %16] [%8, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [2, %17] [%8, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_5 = memref.subview %alloc[0, 0] [2, %16] [%8, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[?, 1]>>
      %subview_6 = memref.subview %alloc[0, %16] [2, %17] [%8, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[?, 1], offset: ?>>
      memref.copy %subview, %subview_5 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, 1]>>
      memref.copy %subview_4, %subview_6 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, 1], offset: ?>>
      %18 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      memref.tensor_store %18, %arg14 : memref<4x4xf32, strided<[?, ?], offset: ?>>
      %19 = arith.index_cast %11 : i32 to index
      %20 = arith.addi %arg15, %19 : index
      %21 = arith.addi %20, %c6 : index
      %22 = arith.remsi %21, %14 : index
      %23 = arith.subi %21, %22 : index
      %24 = arith.addi %22, %c4 : index
      %25 = arith.minsi %24, %14 : index
      %26 = arith.subi %25, %22 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%c4, %26], strides: [%26, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %27 = arith.subi %c4, %26 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [%c4, %27], strides: [%27, %c1] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %28 = arith.addi %arg16, %c4 : index
      %29 = arith.addi %28, %arg17 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [4, 4], strides: [%10, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      scf.yield %reinterpret_cast_7, %reinterpret_cast_9, %20, %29, %c0, %reinterpret_cast_8 : memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>
    }
    return
  }
}

