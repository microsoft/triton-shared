// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @_layer_norm_bwd_dwdb_0123456(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32, %arg5: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %cst : f32 -> tensor<256x256xf32>
    %6 = tt.splat %arg4 : i32 -> tensor<256x1xi32>
    %7 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %8 = tt.splat %arg5 : i32 -> tensor<1x256xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1x256xi32>
    %10 = tt.broadcast %9 : tensor<1x256xi1> -> tensor<256x256xi1>
    %11 = tt.splat %arg5 : i32 -> tensor<256x1xi32>
    %12 = tt.broadcast %7 : tensor<1x256xi32> -> tensor<256x256xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %15:2 = scf.for %arg6 = %c0_i32 to %arg4 step %c256_i32 iter_args(%arg7 = %5, %arg8 = %5) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
      %24 = tt.splat %arg6 : i32 -> tensor<256xi32>
      %25 = arith.addi %24, %2 : tensor<256xi32>
      %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
      %27 = arith.cmpi slt, %26, %6 : tensor<256x1xi32>
      %28 = tt.broadcast %27 : tensor<256x1xi1> -> tensor<256x256xi1>
      %29 = arith.andi %28, %10 : tensor<256x256xi1>
      %30 = arith.muli %26, %11 : tensor<256x1xi32>
      %31 = tt.broadcast %30 : tensor<256x1xi32> -> tensor<256x256xi32>
      %32 = arith.addi %31, %12 : tensor<256x256xi32>
      %33 = tt.addptr %13, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %34 = tt.load %33, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %35 = arith.addf %arg7, %34 : tensor<256x256xf32>
      %36 = tt.addptr %14, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %37 = tt.load %36, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %38 = arith.addf %arg8, %37 : tensor<256x256xf32>
      scf.yield %35, %38 : tensor<256x256xf32>, tensor<256x256xf32>
    }
    %16 = "tt.reduce"(%15#0) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %17 = "tt.reduce"(%15#1) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %18 = tt.splat %arg5 : i32 -> tensor<256xi32>
    %19 = arith.cmpi slt, %4, %18 : tensor<256xi32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %21 = tt.addptr %20, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %21, %16, %19 : tensor<256x!tt.ptr<f32>>
    %22 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %23 = tt.addptr %22, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %23, %17, %19 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK:             %[[PTR:.*]] = tts.make_gather_scatter_tptr %arg0 to sizes: [65536] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_37:.*]] gather_scatter_mask: %[[VAL_38:.*]], strides: [1], offsets: [0] : tensor<65536xi32> tensor<65536xi1> <f32> to !tt.ptr<tensor<65536xf32>>
// CHECK:             %[[VAL_40:.*]] = "tts.load"(%[[PTR]], %[[VAL_11:.*]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<65536xf32>>, f32) -> tensor<65536xf32>
// CHECK:             %[[PTR2:.*]] = tts.make_gather_scatter_tptr %arg1 to sizes: [65536] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_37]] gather_scatter_mask: %[[VAL_38]], strides: [1], offsets: [0] : tensor<65536xi32> tensor<65536xi1> <f32> to !tt.ptr<tensor<65536xf32>>
// CHECK:             %[[VAL_44:.*]] = "tts.load"(%[[PTR2]], %[[VAL_11]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<65536xf32>>, f32) -> tensor<65536xf32>
// CHECK:           %[[PTR3:.*]] = tts.make_gather_scatter_tptr %arg2 to sizes: [256] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_16:.*]] gather_scatter_mask: %[[VAL_58:.*]], strides: [1], offsets: [0] : tensor<256xi32> tensor<256xi1> <f32> to !tt.ptr<tensor<256xf32>>
// CHECK:           "tts.store"(%[[PTR3]], %{{.*}}) <{static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<256xf32>>, tensor<256xf32>) -> ()
// CHECK:           %[[PTR4:.*]] = tts.make_gather_scatter_tptr %arg3 to sizes: [256] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_16]] gather_scatter_mask: %[[VAL_58]], strides: [1], offsets: [0] : tensor<256xi32> tensor<256xi1> <f32> to !tt.ptr<tensor<256xf32>>
// CHECK:           "tts.store"(%[[PTR4]], %{{.*}}) <{static_mask_dims = array<i64: 0>}> : (!tt.ptr<tensor<256xf32>>, tensor<256xf32>) -> ()
// CHECK-NOT:       tts.make_gather_scatter_tptr