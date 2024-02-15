#map = affine_map<(d0) -> (d0)>
module {
  func.func @rand(%arg0: memref<*xi32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %0 = tensor.empty() : tensor<8xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<8xi32>) {
    ^bb0(%out: i32):
      %4 = linalg.index 0 : index
      %5 = arith.index_cast %4 : index to i32
      linalg.yield %5 : i32
    } -> tensor<8xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<*xi32> to memref<8xi32, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi32>
    memref.copy %reinterpret_cast, %alloc : memref<8xi32, strided<[1]>> to memref<8xi32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8xi32>
    %3 = tt.extern_elementwise %2, %1 {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<*xi32> to memref<8xi32, strided<[1]>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<8xi32>, memref<8xi32, strided<[1]>>) -> ()
    return
  }
}

