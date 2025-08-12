// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK:           %[[VAL_12:.*]] = tts.make_gather_scatter_tptr %arg0 to sizes: [1024] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_9:.*]] gather_scatter_mask: %[[VAL_11:.*]], strides: [1], offsets: [0] : tensor<1024xi32> tensor<1024xi1> <f32> to !tt.ptr<tensor<1024xf32>>
// CHECK:           %[[VAL_13:.*]] = "tts.load"(%[[VAL_12]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<1024xf32>>) -> tensor<1024xf32>
// CHECK:           %[[VAL_14:.*]] = tts.make_gather_scatter_tptr %arg1 to sizes: [1024] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_9]] gather_scatter_mask: %[[VAL_11]], strides: [1], offsets: [0] : tensor<1024xi32> tensor<1024xi1> <f32> to !tt.ptr<tensor<1024xf32>>
// CHECK:           %[[VAL_15:.*]] = "tts.load"(%[[VAL_14]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<1024xf32>>) -> tensor<1024xf32>
// CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_13]], %[[VAL_15]] : tensor<1024xf32>
// CHECK:           %[[VAL_17:.*]] = tts.make_gather_scatter_tptr %arg2 to sizes: [1024] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_9]] gather_scatter_mask: %[[VAL_11]], strides: [1], offsets: [0] : tensor<1024xi32> tensor<1024xi1> <f32> to !tt.ptr<tensor<1024xf32>>
// CHECK:           "tts.store"(%[[VAL_17]], %[[VAL_16]]) <{static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<1024xf32>>, tensor<1024xf32>) -> ()
// CHECK-NOT:       tts.make_gather_scatter_tptr
