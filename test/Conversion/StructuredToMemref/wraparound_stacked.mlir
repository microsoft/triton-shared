// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @wrap_stacked_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %3 = arith.remsi %1, %2 : tensor<4xi32>
    %4 = arith.addi %0, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg5 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg6 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg7 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %21, %cst_0 : tensor<1x4xi32>
    %28 = tt.broadcast %27 : tensor<1x4xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg5, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %26) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
      %32 = tt.load %arg9, %28, %cst : tensor<4x4x!tt.ptr<f32>>
      tt.store %arg10, %32 : tensor<4x4x!tt.ptr<f32>>
      %33 = tt.addptr %arg9, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %34 = tt.addptr %arg10, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %33, %34 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:   func.func @wrap_stacked_masked_loop_01234567(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant -9.900000e+01 : f32
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG4]] : i32 to index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_1]], %[[CONSTANT_4]] : index
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[INDEX_CAST_0]], %[[INDEX_CAST_1]] : index
// CHECK:           %[[INDEX_CAST_2:.*]] = arith.index_cast %[[ARG5]] : i32 to index
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[INDEX_CAST_2]], %[[CONSTANT_5]] : index
// CHECK:           %[[INDEX_CAST_3:.*]] = arith.index_cast %[[ARG6]] : i32 to index
// CHECK:           %[[INDEX_CAST_4:.*]] = arith.index_cast %[[ARG7]] : i32 to index
// CHECK:           %[[MULI_3:.*]] = arith.muli %[[ARG5]], %[[CONSTANT_1]] : i32
// CHECK:           %[[INDEX_CAST_5:.*]] = arith.index_cast %[[MULI_3]] : i32 to index
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[MULI_0]], %[[VAL_2:.*]] = %[[CONSTANT_6]]) -> (index, index)  : i32 {
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: {{\[}}%[[VAL_2]]], sizes: [4, 4], strides: {{\[}}%[[INDEX_CAST_3]], %[[INDEX_CAST_4]]] : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[MULI_2]] : index
// CHECK:             %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[INDEX_CAST_1]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[MULI_1]], %[[REMSI_0]] : index
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ADDI_1]], %[[ADDI_0]] : index
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[INDEX_CAST_1]] : index
// CHECK:             %[[MINSI_0:.*]] = arith.minsi %[[DIVSI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[ADDI_0]]], sizes: {{\[}}%[[MINSI_0]], 4], strides: {{\[}}%[[INDEX_CAST_1]], %[[INDEX_CAST_2]]] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[CONSTANT_0]], %[[MINSI_0]] : index
// CHECK:             %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[REMSI_0]]], sizes: {{\[}}%[[SUBI_1]], 4], strides: {{\[}}%[[INDEX_CAST_1]], %[[INDEX_CAST_2]]] : memref<*xf32> to memref<?x4xf32, strided<[?, ?], offset: ?>>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK:             linalg.fill ins(%[[CONSTANT_7]] : f32) outs(%[[ALLOC_0]] : memref<4x4xf32>)
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[REINTERPRET_CAST_1]][0, 0] {{\[}}%[[MINSI_0]], 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK:             %[[SUBVIEW_1:.*]] = memref.subview %[[REINTERPRET_CAST_2]][0, 0] {{\[}}%[[SUBI_1]], 3] [1, 1] : memref<?x4xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK:             %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_0]][0, 0] {{\[}}%[[MINSI_0]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
// CHECK:             %[[SUBVIEW_3:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[MINSI_0]], 0] {{\[}}%[[SUBI_1]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[SUBVIEW_0]], %[[SUBVIEW_2]] : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
// CHECK:             memref.copy %[[SUBVIEW_1]], %[[SUBVIEW_3]] : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_0]] : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[INDEX_CAST_5]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_2]], %[[INDEX_CAST_5]] : index
// CHECK:             scf.yield %[[ADDI_2]], %[[ADDI_3]] : index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }