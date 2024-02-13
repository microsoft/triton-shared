module {
  func.func @reduce_kernel_2d_0d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    %0 = arith.index_cast %arg4 : i32 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast : memref<1xf32> -> memref<f32>, index, index, index
    %reinterpret_cast_0 = memref.reinterpret_cast %base_buffer to offset: [%0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
    %1 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %reinterpret_cast_0) -> (memref<1xf32, strided<[1], offset: ?>>)  : i32 {
      %2 = arith.sitofp %arg7 : i32 to f32
      %3 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %arg8) -> (memref<1xf32, strided<[1], offset: ?>>)  : i32 {
        affine.store %2, %arg10[0] : memref<1xf32, strided<[1], offset: ?>>
        %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %arg10 : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
        %4 = arith.addi %offset_2, %c1 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%4], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
        scf.yield %reinterpret_cast_5 : memref<1xf32, strided<[1], offset: ?>>
      }
      scf.yield %3 : memref<1xf32, strided<[1], offset: ?>>
    }
    return
  }
}

