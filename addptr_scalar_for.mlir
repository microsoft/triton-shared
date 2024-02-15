#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
      %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg12 : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
      %11 = arith.addi %offset_4, %arg11 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [%11], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
    return
  }
}

