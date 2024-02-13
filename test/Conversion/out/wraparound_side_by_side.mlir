module {
  func.func @wrap_side_by_side_masked_loop_01234567(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c4 = arith.constant 4 : index
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.muli %0, %c2 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.muli %3, %c6 : index
    %5 = arith.muli %2, %3 : index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.muli %arg4, %c4_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.muli %arg5, %c4_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12:2 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %c0) -> (index, index)  : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%arg16], sizes: [4, 4], strides: [%6, %7] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      %13 = arith.addi %arg15, %4 : index
      %14 = arith.remsi %13, %5 : index
      %15 = arith.subi %13, %14 : index
      %16 = arith.addi %14, %c4 : index
      %17 = arith.minsi %16, %5 : index
      %18 = arith.subi %17, %14 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [%c4, %18], strides: [%0, %3] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %19 = arith.subi %c4, %18 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%15], sizes: [%c4, %19], strides: [%0, %3] : memref<*xf32> to memref<4x?xf32, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %20 = arith.minsi %18, %c4 : index
      %21 = arith.subi %c4, %20 : index
      %subview = memref.subview %reinterpret_cast_0[0, 0] [2, %20] [1, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [2, %21] [1, 1] : memref<4x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [2, %20] [1, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[0, %20] [2, %21] [1, 1] : memref<4x4xf32> to memref<2x?xf32, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_3 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[4, 1]>>
      memref.copy %subview_2, %subview_4 : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<2x?xf32, strided<[4, 1], offset: ?>>
      %22 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      bufferization.materialize_in_destination %22 in writable %reinterpret_cast : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
      %23 = arith.addi %arg15, %9 : index
      %24 = arith.addi %arg16, %11 : index
      scf.yield %23, %24 : index, index
    }
    return
  }
}

