// RUN: triton-shared-opt --structured-to-memref --split-input-file %s | FileCheck %s

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

// ---

module {
  tt.func public @add_ptr_into_make_block_ptr(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c32768_i64 : i64
    %3 = tt.addptr %arg0, %2 : !tt.ptr<bf16>, i64
    %4 = tt.make_tensor_ptr %3, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %5 = tt.addptr %arg1, %2 : !tt.ptr<bf16>, i64
    %6 = tt.make_tensor_ptr %5, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %7 = tt.load %4 : !tt.ptr<tensor<512x64xbf16>>
    tt.store %6, %7 : !tt.ptr<tensor<512x64xbf16>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @add_ptr_into_make_block_ptr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:      [[CST_64_32768_:%.+]] = arith.constant 32768 : i64
// CHECK-DAG:      [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:      [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:      [[VAR_0_:%.+]] = arith.extsi [[PARAM_5_]] : i32 to i64
// CHECK:          [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_64_32768_]] : i64
// CHECK:          [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i64 to index
// CHECK-DAG:      [[RCAST_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [[[VAR_2_]]], sizes: [512, 64], strides: [[[CST_64_]], [[CST_1_]]] : memref<*xbf16> to memref<512x64xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:      [[RCAST_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [[[VAR_2_]]], sizes: [512, 64], strides: [[[CST_64_]], [[CST_1_]]] : memref<*xbf16> to memref<512x64xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:      [[ALLOC_:%.+]] = memref.alloc() : memref<512x64xbf16>
// CHECK:          memref.copy [[RCAST_]], [[ALLOC_]] : memref<512x64xbf16, strided<[?, ?], offset: ?>> to memref<512x64xbf16>
// CHECK:          [[VAR_3_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<512x64xbf16>
// CHECK:          bufferization.materialize_in_destination [[VAR_3_]] in writable [[RCAST_0_]] : (tensor<512x64xbf16>, memref<512x64xbf16, strided<[?, ?], offset: ?>>) -> ()
// CHECK:          return
// CHECK:        }
