module {
  func.func @wrap_stacked_masked_loop_01234567(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %cst = arith.constant -9.900000e+01 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %1, %c2 : index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.muli %3, %c3 : index
    %5 = arith.addi %2, %4 : index
    %6 = arith.remsi %5, %1 : index
    %7 = arith.muli %0, %1 : index
    %8 = arith.addi %7, %6 : index
    %9 = arith.subi %8, %5 : index
    %10 = arith.divsi %9, %1 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [%10, %c4], strides: [%1, %3] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
    %11 = arith.subi %c4, %10 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [%11, %c4], strides: [%1, %3] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
    %12 = arith.index_cast %arg6 : i32 to index
    %13 = arith.index_cast %arg7 : i32 to index
    %14 = arith.muli %arg5, %c4_i32 : i32
    %15 = arith.index_cast %arg2 : i32 to index
    %16 = arith.index_cast %arg4 : i32 to index
    %17 = arith.muli %16, %c2 : index
    %18 = arith.index_cast %arg5 : i32 to index
    %19 = arith.muli %18, %c3 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [4, 4], strides: [%12, %13] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %20:6 = scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg15 = %reinterpret_cast, %arg16 = %reinterpret_cast_1, %arg17 = %17, %arg18 = %c0, %arg19 = %c0, %arg20 = %reinterpret_cast_0) -> (memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>)  : i32 {
      %alloc = memref.alloc() : memref<4x4xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
      %dim = memref.dim %arg15, %c0 : memref<?x4xf32, strided<[?, ?], offset: ?>>
      %21 = arith.minsi %dim, %c4 : index
      %22 = arith.subi %c4, %21 : index
      %subview = memref.subview %arg15[0, 0] [%21, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_2 = memref.subview %arg20[0, 0] [%22, 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%21, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
      %subview_4 = memref.subview %alloc[%21, 0] [%22, 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_3 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
      memref.copy %subview_2, %subview_4 : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
      %23 = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32>
      bufferization.materialize_in_destination %23 in writable %arg16 : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
      %24 = arith.index_cast %14 : i32 to index
      %25 = arith.addi %arg17, %24 : index
      %26 = arith.addi %25, %19 : index
      %27 = arith.remsi %26, %16 : index
      %28 = arith.muli %15, %16 : index
      %29 = arith.addi %28, %27 : index
      %30 = arith.subi %29, %26 : index
      %31 = arith.divsi %30, %16 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [%31, %c4], strides: [%16, %18] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %32 = arith.subi %c4, %31 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [%32, %c4], strides: [%16, %18] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
      %33 = arith.index_cast %14 : i32 to index
      %34 = arith.addi %arg18, %33 : index
      %35 = arith.addi %34, %arg19 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%35], sizes: [4, 4], strides: [%12, %13] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
      scf.yield %reinterpret_cast_5, %reinterpret_cast_7, %25, %35, %c0, %reinterpret_cast_6 : memref<?x4xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<?x4xf32, strided<[?, ?], offset: ?>>
    }
    return
  }
}

