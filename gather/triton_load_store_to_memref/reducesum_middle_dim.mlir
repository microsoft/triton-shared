module {
  func.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: tensor<32x16x!tt.ptr<bf16>>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<32x16xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<32x16xi32>) -> tensor<32x16xi32>
    %2 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<bf16> to memref<*xbf16>
    %reinterpret_cast = memref.reinterpret_cast %2 to offset: [0], sizes: [32, 256, 16], strides: [%c256, 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[?, 1, 1]>>
    %alloc = memref.alloc() : memref<32x256x16xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<32x256x16xbf16, strided<[?, 1, 1]>> to memref<32x256x16xbf16>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<32x256x16xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %4 = tensor.empty() : tensor<32x16xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%4 : tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %reduced = linalg.reduce ins(%3 : tensor<32x256x16xbf16>) outs(%5 : tensor<32x16xbf16>) dimensions = [1]
      (%in: bf16, %init: bf16) {
        %7 = arith.addf %in, %init : bf16
        linalg.yield %7 : bf16
      }
    %6 = "tts.make_unstructured_tptr"(%arg2, %1) : (tensor<32x16x!tt.ptr<bf16>>, tensor<32x16xi32>) -> tensor<32x16x!tt.ptr<bf16>>
    tt.store %6, %reduced : tensor<32x16x!tt.ptr<bf16>>
    return
  }
}