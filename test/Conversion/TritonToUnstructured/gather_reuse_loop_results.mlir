// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @nested_use_same_level_loop_result(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
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
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
      }
      %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %29 = tt.addptr %arg9, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.addptr %29, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
        %31 = tt.addptr %30, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %32 = tt.addptr %27, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %25 = tt.addptr %24#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %25, %24#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @nested_use_same_level_loop_result(
// CHECK-SAME:                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                                      %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                                      %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                                      %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<0> : tensor<1xindex>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tt.expand_dims %[[VAL_8]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK:           %[[VAL_10:.*]] = tt.splat %[[VAL_2]] : i32 -> tensor<2x1xi32>
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_10]] : tensor<2x1xi32>
// CHECK:           %[[VAL_12:.*]] = tt.expand_dims %[[VAL_8]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK:           %[[VAL_13:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<1x2xi32>
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : tensor<1x2xi32>
// CHECK:           %[[VAL_15:.*]] = tt.broadcast %[[VAL_11]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_16:.*]] = tt.broadcast %[[VAL_14]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : tensor<2x2xi32>
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_3]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_18]] : i32 -> tensor<2x2xi32>
// CHECK:           %[[VAL_20:.*]]:2 = scf.for %[[VAL_21:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_22:.*]] = %[[VAL_17]], %[[VAL_23:.*]] = %[[VAL_17]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK:             %[[VAL_24:.*]] = scf.for %[[VAL_25:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_26:.*]] = %[[VAL_22]]) -> (tensor<2x2xi32>)  : i32 {
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               scf.yield %[[VAL_27]] : tensor<2x2xi32>
// CHECK:             }
// CHECK:             %[[VAL_28:.*]]:2 = scf.for %[[VAL_29:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_30:.*]] = %[[VAL_24]], %[[VAL_31:.*]] = %[[VAL_23]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK:               %[[VAL_32:.*]] = tensor.reshape %[[VAL_30]](%[[VAL_4]]) : (tensor<2x2xi32>, tensor<1xindex>) -> tensor<4xi32>
// CHECK:               %[[VAL_33:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_32]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               %[[VAL_34:.*]] = "tts.load"(%[[VAL_33]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>) -> tensor<4xf32>
// CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_30]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               %[[VAL_36:.*]] = tensor.reshape %[[VAL_35]](%[[VAL_4]]) : (tensor<2x2xi32>, tensor<1xindex>) -> tensor<4xi32>
// CHECK:               %[[VAL_37:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_36]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               %[[VAL_38:.*]] = "tts.load"(%[[VAL_37]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>) -> tensor<4xf32>
// CHECK:               %[[VAL_39:.*]] = tensor.reshape %[[VAL_31]](%[[VAL_4]]) : (tensor<2x2xi32>, tensor<1xindex>) -> tensor<4xi32>
// CHECK:               %[[VAL_40:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_39]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               "tts.store"(%[[VAL_40]], %[[VAL_34]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>, tensor<4xf32>) -> ()
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_31]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               %[[VAL_43:.*]] = tensor.reshape %[[VAL_42]](%[[VAL_4]]) : (tensor<2x2xi32>, tensor<1xindex>) -> tensor<4xi32>
// CHECK:               %[[VAL_44:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_43]], strides: [1], offsets: [0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4xf32>>
// CHECK:               "tts.store"(%[[VAL_44]], %[[VAL_38]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4xf32>>, tensor<4xf32>) -> ()
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_42]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_35]], %[[VAL_19]] : tensor<2x2xi32>
// CHECK:               scf.yield %[[VAL_46]], %[[VAL_45]] : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:             }
// CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_48:.*]]#0, %[[VAL_19]] : tensor<2x2xi32>
// CHECK:             scf.yield %[[VAL_47]], %[[VAL_48]]#1 : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
