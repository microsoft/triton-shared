// RUN: triton-shared-opt --structured-to-memref --unstructured-to-memref %s | FileCheck %s
// Check that the reinterpret_cast lowered from tts.make_tptr uses the correct offset from tts.make_unstructured_tptr

module {
  tt.func public @add_ptr_into_make_block_ptr(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32768_i64 = arith.constant 32768 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c32768_i64 : i64
    %3 = "tts.make_unstructured_tptr"(%arg0, %2) : (!tt.ptr<bf16>, i64) -> !tt.ptr<bf16>
    %4 = tts.make_tptr %3 to sizes: [512, 64], strides: [%c64, %c1], offsets: [%c0, %c0], shape: [%c512, %c64], order: [1, 0] : <bf16> to !tt.ptr<tensor<512x64xbf16>>
    %5 = "tts.make_unstructured_tptr"(%arg1, %2) : (!tt.ptr<bf16>, i64) -> !tt.ptr<bf16>
    %6 = tts.make_tptr %5 to sizes: [512, 64], strides: [%c64, %c1], offsets: [%c0, %c0], shape: [%c512, %c64], order: [1, 0] : <bf16> to !tt.ptr<tensor<512x64xbf16>>
    %7 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<512x64xbf16>>) -> tensor<512x64xbf16>
    "tts.store"(%6, %7) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<512x64xbf16>>, tensor<512x64xbf16>) -> ()
    tt.return
  }
}

// CHECK: module {
// CHECK:   tt.func public @add_ptr_into_make_block_ptr(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c64 = arith.constant 64 : index
// CHECK:     %c32768_i64 = arith.constant 32768 : i64
// CHECK:     %0 = tt.get_program_id x : i32
// CHECK:     %1 = arith.extsi %0 : i32 to i64
// CHECK:     %2 = arith.muli %1, %c32768_i64 : i64
// CHECK:     %3 = "tts.make_unstructured_tptr"(%arg0, %2) : (!tt.ptr<bf16>, i64) -> !tt.ptr<bf16>
// CHECK:     %4 = builtin.unrealized_conversion_cast %3 : !tt.ptr<bf16> to memref<*xbf16>
// CHECK:     %5 = arith.index_cast %2 : i64 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %4 to offset: [%5], sizes: [512, 64], strides: [%c64, %c1] : memref<*xbf16> to memref<512x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %6 = "tts.make_unstructured_tptr"(%arg1, %2) : (!tt.ptr<bf16>, i64) -> !tt.ptr<bf16>
// CHECK:     %7 = builtin.unrealized_conversion_cast %6 : !tt.ptr<bf16> to memref<*xbf16>
// CHECK:     %8 = arith.index_cast %2 : i64 to index
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %7 to offset: [%8], sizes: [512, 64], strides: [%c64, %c1] : memref<*xbf16> to memref<512x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %alloc = memref.alloc() : memref<512x64xbf16>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<512x64xbf16, strided<[?, ?], offset: ?>> to memref<512x64xbf16>
// CHECK:     %9 = bufferization.to_tensor %alloc restrict writable : memref<512x64xbf16>
// CHECK:     bufferization.materialize_in_destination %9 in writable %reinterpret_cast_0 : (tensor<512x64xbf16>, memref<512x64xbf16, strided<[?, ?], offset: ?>>) -> ()
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
