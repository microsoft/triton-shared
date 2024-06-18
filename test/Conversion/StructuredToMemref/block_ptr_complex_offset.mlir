// RUN: triton-shared-opt --structured-to-memref %s | FileCheck %s

module {
  func.func @fused_attention_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: i64, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) -> tensor<128x128xbf16> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = arith.remsi %arg8, %arg3 : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %arg2 : i64
    %3 = tt.addptr %arg0, %2 : !tt.ptr<bf16>, i64
    %4 = tts.make_tptr %3 to sizes: [128, 128], strides: [%c128, %c1], offsets: [%c0, %c0], shape: [%c128, %c128], order: [1, 0] : <bf16> to !tt.ptr<tensor<128x128xbf16>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<128x128xbf16>>) -> tensor<128x128xbf16>
    return %5 : tensor<128x128xbf16>
  }
}

// CHECK: module {
// CHECK:   func.func @fused_attention_fwd_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i64, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) -> tensor<128x128xbf16> {
// CHECK-DAG:     %c128 = arith.constant 128 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK:     %0 = arith.remsi %arg8, %arg3 : i32
// CHECK:     %1 = arith.extsi %0 : i32 to i64
// CHECK:     %2 = arith.muli %1, %arg2 : i64
// CHECK:     %3 = arith.index_cast %2 : i64 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [128, 128], strides: [%c128, %c1] : memref<*xbf16> to memref<128x128xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %alloc = memref.alloc() : memref<128x128xbf16>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<128x128xbf16, strided<[?, ?], offset: ?>> to memref<128x128xbf16>
// CHECK:     %4 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
// CHECK:     return %4 : tensor<128x128xbf16>
// CHECK:   }
// CHECK: }
