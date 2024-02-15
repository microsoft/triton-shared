module {
  func.func @wrap_side_by_side_masked_loop_01234567(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.muli %0, %c2 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.muli %3, %c6 : index
    %5 = arith.addi %1, %4 : index
    %6 = arith.remsi %5, %2 : index
    %7 = arith.subi %5, %6 : index
    %8 = arith.addi %6, %c4 : index
    %9 = arith.minsi %8, %2 : index
    %10 = arith.subi %9, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [%c4, %10], strides: [%0, %3] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %11 = arith.subi %c4, %10 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [%c4, %11], strides: [%0, %3] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %12 = arith.index_cast %arg6 : i32 to index
    %13 = arith.index_cast %arg7 : i32 to index
    %14 = arith.muli %arg4, %c4_i32 : i32
    %15 = arith.muli %arg5, %c4_i32 : i32
    %16 = arith.index_cast %arg4 : i32 to index
    %17 = arith.muli %16, %c2 : index
    %18 = arith.index_cast %arg3 : i32 to index
    %19 = arith.index_cast %arg5 : i32 to index
    %20 = arith.muli %19, %c6 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [4, 4], strides: [%12, %13] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %21:6 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %reinterpret_cast, %arg16 = %reinterpret_cast_1, %arg17 = %17, %arg18 = %c0, %arg19 = %c0, %arg20 = %reinterpret_cast_0) -> (memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>)  : i32 {
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %dim = memref.dim %arg15, %c1 : memref<4x?xf32, strided<[?, ?], offset: ?>>
      %22 = arith.minsi %dim, %c4 : index
      %23 = arith.subi %c4, %22 : index
      %subview = memref.subview %arg15[0, 0] [2, %22] [1, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_2 = memref.subview %arg20[0, 0] [2, %23] [1, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [2, %22] [1, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[0, %22] [2, %23] [1, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_3 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[4, 1]>>
      memref.copy %subview_2, %subview_4 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[4, 1], offset: ?>>
      %24 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      bufferization.materialize_in_destination %24 in writable %arg16 : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
      %25 = arith.index_cast %14 : i32 to index
      %26 = arith.addi %arg17, %25 : index
      %27 = arith.addi %26, %20 : index
      %28 = arith.remsi %27, %18 : index
      %29 = arith.subi %27, %28 : index
      %30 = arith.addi %28, %c4 : index
      %31 = arith.minsi %30, %18 : index
      %32 = arith.subi %31, %28 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [%c4, %32], strides: [%16, %19] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %33 = arith.subi %c4, %32 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [%c4, %33], strides: [%16, %19] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %34 = arith.index_cast %15 : i32 to index
      %35 = arith.addi %arg18, %34 : index
      %36 = arith.addi %35, %arg19 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [4, 4], strides: [%12, %13] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      scf.yield %reinterpret_cast_5, %reinterpret_cast_7, %26, %36, %c0, %reinterpret_cast_6 : memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>
    }
    return
  }
}

