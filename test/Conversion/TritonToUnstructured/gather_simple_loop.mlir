// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_0 : tensor<64xi32>
      %5 = arith.addi %arg2, %c5_i32 : i32
      %6 = arith.remsi %5, %c64_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<64xi32>
      %8 = arith.addi %4, %7 : tensor<64xi32>
      %9 = tt.addptr %1, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
      %11 = tt.addptr %2, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %11, %10 : tensor<64x!tt.ptr<f32>>
      %12 = arith.addi %8, %cst : tensor<64xi32>
      %13 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %12, %13 : tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK:           %[[VAL_10:.*]]:2 = scf.for %[[VAL_11:.*]] = %[[VAL_2:.*]] to %[[VAL_4:.*]] step %[[VAL_3:.*]] iter_args(%[[VAL_12:.*]] = %[[VAL_9:.*]], %[[VAL_13:.*]] = %[[VAL_9]]) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:             %[[VAL_14:.*]] = arith.divsi %[[VAL_12]], %[[VAL_8:.*]] : tensor<64xi32>
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_11]], %[[VAL_7:.*]] : i32
// CHECK:             %[[VAL_16:.*]] = arith.remsi %[[VAL_15]], %[[VAL_6:.*]] : i32
// CHECK:             %[[VAL_17:.*]] = tt.splat %[[VAL_16]] : i32 -> tensor<64xi32>
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_14]], %[[VAL_17]] : tensor<64xi32>
// CHECK:             %[[VAL_19:.*]] = tts.make_gather_scatter_tptr %[[VAL_0:.*]] to sizes: [64] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_18]], strides: [1], offsets: [0] : tensor<64xi32>  <f32> to !tt.ptr<tensor<64xf32>>
// CHECK:             %[[VAL_20:.*]] = "tts.load"(%[[VAL_19]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<64xf32>>) -> tensor<64xf32>
// CHECK:             %[[VAL_21:.*]] = tts.make_gather_scatter_tptr %[[VAL_1:.*]] to sizes: [64] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_13]], strides: [1], offsets: [0] : tensor<64xi32>  <f32> to !tt.ptr<tensor<64xf32>>
// CHECK:             "tts.store"(%[[VAL_21]], %[[VAL_20]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<64xf32>>, tensor<64xf32>) -> ()
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_18]], %[[VAL_5:.*]] : tensor<64xi32>
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_13]], %[[VAL_5]] : tensor<64xi32>
// CHECK:             scf.yield %[[VAL_22]], %[[VAL_23]] : tensor<64xi32>, tensor<64xi32>
// CHECK:           }
