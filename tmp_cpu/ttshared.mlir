module {
  func.func @wrap_stacked_masked_loop_0d1d2345c67c(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c7 = arith.constant 7 : index
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %1, %c2 : index
    %3 = arith.addi %2, %c3 : index
    %4 = arith.remsi %3, %1 : index
    %5 = arith.muli %0, %1 : index
    %6 = arith.addi %5, %4 : index
    %7 = arith.subi %6, %3 : index
    %8 = arith.divsi %7, %1 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [%8, %c4], strides: [%1, %c1] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
    %9 = arith.subi %c4, %8 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [%9, %c4], strides: [%1, %c1] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
    %10 = arith.index_cast %arg5 : i32 to index
    %11 = arith.index_cast %arg2 : i32 to index
    %12 = arith.index_cast %arg4 : i32 to index
    %13 = arith.muli %12, %c2 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [4, 4], strides: [%10, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %14:6 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %reinterpret_cast, %arg14 = %reinterpret_cast_1, %arg15 = %13, %arg16 = %c0, %arg17 = %c0, %arg18 = %reinterpret_cast_0) -> (memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>)  : i32 {
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %dim = memref.dim %arg13, %c0 : memref<?x4xf32, strided<[?, ?], offset: ?>>
      %15 = arith.minsi %dim, %c4 : index
      %16 = arith.subi %c4, %15 : index
      %subview = memref.subview %arg13[0, 0] [%15, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_2 = memref.subview %arg18[0, 0] [%16, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%15, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[%15, 0] [%16, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_3 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
      memref.copy %subview_2, %subview_4 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      %17 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      memref.tensor_store %17, %arg14 : memref<4x4xf32, strided<[?, ?], offset: ?>>
      %18 = arith.addi %arg15, %c4 : index
      %19 = arith.addi %arg15, %c7 : index
      %20 = arith.remsi %19, %12 : index
      %21 = arith.muli %11, %12 : index
      %22 = arith.addi %21, %20 : index
      %23 = arith.subi %22, %19 : index
      %24 = arith.divsi %23, %12 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%19], sizes: [%24, %c4], strides: [%12, %c1] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %25 = arith.subi %c4, %24 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%25, %c4], strides: [%12, %c1] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %26 = arith.addi %arg16, %c4 : index
      %27 = arith.addi %26, %arg17 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [4, 4], strides: [%10, %c1] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      scf.yield %reinterpret_cast_5, %reinterpret_cast_7, %18, %27, %c0, %reinterpret_cast_6 : memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>
    }
    return
  }
}

