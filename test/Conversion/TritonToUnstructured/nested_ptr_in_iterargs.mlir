// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<3> : tensor<2x2xi32>
    %cst_0 = arith.constant dense<1> : tensor<2x2xi32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg2, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.addptr %arg5, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %20 = tt.addptr %arg6, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %19, %arg9 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.addptr %arg9, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %27, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %22 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %23 = tt.addptr %22, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %25 = tt.addptr %24, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @nested2_complex_body(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                         %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                         %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<3> : tensor<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<1> : tensor<2x2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_10:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[VAL_11:.*]] = tt.expand_dims %[[VAL_10]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_2]] : i32 -> tensor<2x1xi32>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_11]], %[[VAL_12]] : tensor<2x1xi32>
// CHECK:           %[[VAL_14:.*]] = tt.expand_dims %[[VAL_10]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK:           %[[VAL_15:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<1x2xi32>
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_14]], %[[VAL_15]] : tensor<1x2xi32>
// CHECK:           %[[VAL_17:.*]] = tt.broadcast %[[VAL_13]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_18:.*]] = tt.broadcast %[[VAL_16]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : tensor<2x2xi32>
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_2]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_21:.*]] = tt.splat %[[VAL_20]] : i32 -> tensor<2x2xi32>
// CHECK:           %[[VAL_22:.*]]:2 = scf.for %[[VAL_23:.*]] = %[[VAL_5]] to %[[VAL_9]] step %[[VAL_6]] iter_args(%[[VAL_24:.*]] = %[[VAL_19]], %[[VAL_25:.*]] = %[[VAL_19]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_24]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:             %[[VAL_28:.*]]:2 = scf.for %[[VAL_29:.*]] = %[[VAL_5]] to %[[VAL_9]] step %[[VAL_6]] iter_args(%[[VAL_30:.*]] = %[[VAL_26]], %[[VAL_31:.*]] = %[[VAL_27]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK:               %[[VAL_32:.*]] = tensor.collapse_shape %[[VAL_30]] {{.*}}0, 1{{.*}} : tensor<2x2xi32> into tensor<4xi32>
// CHECK:               %[[VAL_33:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_32]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               %[[VAL_34:.*]] = "tts.load"(%[[VAL_33]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>) -> tensor<4xf32>
// CHECK:               %[[VAL_35:.*]] = tensor.collapse_shape %[[VAL_31]] {{.*}}0, 1{{.*}} : tensor<2x2xi32> into tensor<4xi32>
// CHECK:               %[[VAL_36:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_35]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               "tts.store"(%[[VAL_36]], %[[VAL_34]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>, tensor<4xf32>) -> ()
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_30]], %[[VAL_7]] : tensor<2x2xi32>
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_31]], %[[VAL_7]] : tensor<2x2xi32>
// CHECK:               scf.yield %[[VAL_37]], %[[VAL_38]] : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_24]], %[[VAL_21]] : tensor<2x2xi32>
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_25]], %[[VAL_21]] : tensor<2x2xi32>
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:             scf.yield %[[VAL_40]], %[[VAL_42]] : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
