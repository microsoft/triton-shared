module {
  func.func @minmax_olt(%arg0: memref<*xf32>, %arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.minimumf %arg1, %arg2 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    affine.store %0, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}


// -----
module {
  func.func @minmax_ole(%arg0: memref<*xf32>, %arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.minimumf %arg1, %arg2 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    affine.store %0, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}


// -----
module {
  func.func @minmax_ogt(%arg0: memref<*xf32>, %arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.maximumf %arg1, %arg2 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    affine.store %0, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}


// -----
module {
  func.func @minmax_oge(%arg0: memref<*xf32>, %arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.maximumf %arg1, %arg2 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32>
    affine.store %0, %reinterpret_cast[0] : memref<1xf32>
    return
  }
}

