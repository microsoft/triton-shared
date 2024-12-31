module {
  tt.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: tensor<256x16x!tt.ptr<bf16>>) {
    %c256 = arith.constant 256 : index
    %cst = arith.constant dense<0> : tensor<256x16xi64>
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<bf16> to memref<*xbf16>
    %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [32, 256, 16], strides: [%c256, 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[?, 1, 1]>>
    %alloc = memref.alloc() : memref<32x256x16xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<32x256x16xbf16, strided<[?, 1, 1]>> to memref<32x256x16xbf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<32x256x16xbf16>
    %2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({
    ^bb0(%arg2: bf16, %arg3: bf16):
      %4 = arith.cmpf ogt, %arg2, %arg3 : bf16
      %5 = arith.select %4, %arg2, %arg3 : bf16
      tt.reduce.return %5 : bf16
    }) : (tensor<32x256x16xbf16>) -> tensor<256x16xbf16>
    %3 = "tts.make_unstructured_tptr"(%arg1, %cst) : (tensor<256x16x!tt.ptr<bf16>>, tensor<256x16xi64>) -> tensor<256x16x!tt.ptr<bf16>>
    tt.store %3, %2 : tensor<256x16x!tt.ptr<bf16>>
    tt.return
  }
}