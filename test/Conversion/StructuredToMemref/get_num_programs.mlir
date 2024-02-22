// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s

module {
  tt.func public @num_programs(%arg0: !tt.ptr<i32>) {
    %0 = tt.get_num_programs {axis = 0 : i32} : i32
    %1 = tt.get_num_programs {axis = 1 : i32} : i32
    %2 = tt.get_num_programs {axis = 2 : i32} : i32
    %3 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %4 = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>
    %5 = tt.make_range {end = 3 : i32, start = 2 : i32} : tensor<1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %7 = tt.addptr %6, %3 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %8 = tt.splat %0 : i32 -> tensor<1xi32>
    tt.store %7, %8 {cache = 1 : i32, evict = 1 : i32} : tensor<1xi32>
    %9 = tt.addptr %6, %4 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %10 = tt.splat %1 : i32 -> tensor<1xi32>
    tt.store %9, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<1xi32>
    %11 = tt.addptr %6, %5 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %12 = tt.splat %2 : i32 -> tensor<1xi32>
    tt.store %11, %12 {cache = 1 : i32, evict = 1 : i32} : tensor<1xi32>
    tt.return
  }
}

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @num_programs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<1xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[PARAM_1_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_1_]] in writable [[VAR_reinterpret_cast_]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [1], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: 1>>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<1xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_2_]] : i32) outs([[VAR_2_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_3_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1xi32>, memref<1xi32, strided<[1], offset: 1>>) -> ()
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [2], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: 2>>
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<1xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.fill ins([[PARAM_3_]] : i32) outs([[VAR_4_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:           bufferization.materialize_in_destination [[VAR_5_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<1xi32>, memref<1xi32, strided<[1], offset: 2>>) -> ()
// CHECK:           return
// CHECK:         }
