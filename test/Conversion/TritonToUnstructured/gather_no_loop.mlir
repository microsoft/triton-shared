// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %2 = arith.addi %1, %cst : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.load %4 : tensor<64x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %7, %5 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK:           %[[PTR:.*]] = tts.make_gather_scatter_tptr %arg0 to sizes: [64] gather_scatter_dim: 0 gather_scatter_offset: {{.*}}, strides: [1], offsets: [0] : tensor<64xi32>  <f32> to !tt.ptr<tensor<64xf32>>
// CHECK:           %[[VAL_8:.*]] = "tts.load"(%[[PTR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<64xf32>>) -> tensor<64xf32>
// CHECK:           %[[PTR2:.*]] = tts.make_gather_scatter_tptr %arg1 to sizes: [64] gather_scatter_dim: 0 gather_scatter_offset: {{.*}}, strides: [1], offsets: [0] : tensor<64xi32>  <f32> to !tt.ptr<tensor<64xf32>>
// CHECK:           "tts.store"(%[[PTR2]], %[[VAL_8]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<64xf32>>, tensor<64xf32>) -> ()
