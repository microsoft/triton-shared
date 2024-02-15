module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16>
    %0 = arith.index_cast %arg2 : i32 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast_0 : memref<1xbf16> -> memref<bf16>, index, index, index
    %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer to offset: [%0], sizes: [1], strides: [1] : memref<bf16> to memref<1xbf16, strided<[1], offset: ?>>
    %1 = arith.index_cast %arg2 : i32 to index
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %reinterpret_cast : memref<1xbf16> -> memref<bf16>, index, index, index
    %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%1], sizes: [1], strides: [1] : memref<bf16> to memref<1xbf16, strided<[1], offset: ?>>
    %2 = affine.load %reinterpret_cast_1[0] : memref<1xbf16, strided<[1], offset: ?>>
    affine.store %2, %reinterpret_cast_6[0] : memref<1xbf16, strided<[1], offset: ?>>
    return
  }
}

