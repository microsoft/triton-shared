// RUN: triton-shared-opt --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s

module {
  func.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %arg4 : !tt.ptr<f32>, i32
    %1 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %0) -> (!tt.ptr<f32>)  : i32 {
      %2 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %arg8) -> (!tt.ptr<f32>)  : i32 {
        %3 = arith.muli %arg7, %arg9 : i32
        %4 = arith.sitofp %3 : i32 to f32
        tt.store %arg10, %4 : !tt.ptr<f32>
        %5 = tt.addptr %arg10, %c1_i32 : !tt.ptr<f32>, i32
        scf.yield %5 : !tt.ptr<f32>
      }
      scf.yield %2 : !tt.ptr<f32>
    }
    return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_1_:%.+]]:2 = scf.for [[VAR_arg7_:%.+]] = [[CST_0_]] to [[CST_8_]] step [[CST_1_1_]] iter_args([[VAR_arg8_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg9_:%.+]] = [[VAR_0_]]) -> (memref<1xf32, strided<[1], offset: ?>>, index)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_1_]] iter_args([[VAR_arg11_:%.+]] = [[VAR_arg8_]], [[VAR_arg12_:%.+]] = [[VAR_arg9_]]) -> (memref<1xf32, strided<[1], offset: ?>>, index)  : i32 {
// CHECK-DAG:           [[VAR_3_:%.+]] = arith.muli [[VAR_arg7_]], [[VAR_arg10_]] : i32
// CHECK:               [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i32 to f32
// CHECK:               affine.store [[VAR_4_]], [[VAR_arg11_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:               [[VAR_5_:%.+]] = arith.addi [[VAR_arg12_]], [[CST_1_]] : index
// CHECK:               [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[VAR_arg11_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: [1], strides: [1] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               scf.yield [[VAR_reinterpret_cast_0_]], [[VAR_5_]] : memref<1xf32, strided<[1], offset: ?>>, index
// CHECK:             }
// CHECK:             scf.yield [[VAR_2_]]#0, [[VAR_2_]]#1 : memref<1xf32, strided<[1], offset: ?>>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
