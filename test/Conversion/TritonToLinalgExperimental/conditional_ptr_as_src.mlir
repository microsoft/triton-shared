// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @simple_cf_into_structured_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %1 = scf.if %0 -> (!tt.ptr<f32>) {
      %9 = arith.muli %arg2, %c2_i32 : i32
      %10 = tt.addptr %arg0, %9 : !tt.ptr<f32>, i32
      scf.yield %10 : !tt.ptr<f32>
    } else {
      %9 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      scf.yield %9 : !tt.ptr<f32>
    }
    %2 = tt.addptr %1, %c6_i32 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %8, %6 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @simple_cf_into_structured_load
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tptr.type_offset f32  : i32
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_cast_:%.+]] = memref.cast [[PARAM_0_]] : memref<*xf32> to memref<1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = tptr.from_memref [[VAR_cast_]] : memref<1xf32> to <#ptr.generic_space>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi eq, [[PARAM_2_]], [[CST_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = scf.if [[VAR_2_]] -> (!ptr.ptr<#ptr.generic_space>) {
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.muli [[PARAM_2_]], [[CST_2_]] : i32
// CHECK:             [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[VAR_0_]] : i32
// CHECK:             [[VAR_8_:%.+]] = tptr.ptradd [[VAR_1_]] [[VAR_7_]] : <#ptr.generic_space>, i32 to <#ptr.generic_space>
// CHECK:             scf.yield [[VAR_8_]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } else {
// CHECK:             [[VAR_6_1_:%.+]] = arith.muli [[PARAM_2_]], [[VAR_0_]] : i32
// CHECK:             [[VAR_7_1_:%.+]] = tptr.ptradd [[VAR_1_]] [[VAR_6_1_]] : <#ptr.generic_space>, i32 to <#ptr.generic_space>
// CHECK:             scf.yield [[VAR_7_1_]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = tptr.to_memref [[VAR_3_]] : <#ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[VAR_4_]] to offset: {{.}}[[CST_6_]]{{.}}, sizes: [4], strides: [1] : memref<1xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_5_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
