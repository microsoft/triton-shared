module {
  func.func @wrap_stacked_masked_loop_01234567(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant -9.900000e+01 : f32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %1, %c2 : index
    %3 = arith.muli %0, %1 : index
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = arith.muli %4, %c3 : index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.muli %arg5, %c4_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.index_cast %8 : i32 to index
    %11:2 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %2, %arg16 = %c0) -> (index, index)  : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%arg16], sizes: [4, 4], strides: [%6, %7] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      %12 = arith.addi %arg15, %5 : index
      %13 = arith.remsi %12, %1 : index
      %14 = arith.addi %3, %13 : index
      %15 = arith.subi %14, %12 : index
      %16 = arith.divsi %15, %1 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [%16, %c4], strides: [%1, %4] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %17 = arith.subi %c4, %16 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [%17, %c4], strides: [%1, %4] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %18 = arith.minsi %16, %c4 : index
      %19 = arith.subi %c4, %18 : index
      %subview = memref.subview %reinterpret_cast_0[0, 0] [%18, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%19, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%18, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[%18, 0] [%19, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_3 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
      memref.copy %subview_2, %subview_4 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      %20 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      bufferization.materialize_in_destination %20 in writable %reinterpret_cast : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
      %21 = arith.addi %arg15, %10 : index
      %22 = arith.addi %arg16, %9 : index
      scf.yield %21, %22 : index, index
    }
    return
  }
}

