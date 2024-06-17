// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<f32>,
        %res : !tt.ptr<f32>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : tensor<512x1xi32> -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : tensor<1x256xi32> -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<f32> -> tensor<512x256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<f32>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 1 : i32} : (tensor<512x256xf32>) -> tensor<512xf32>
    tt.store %19, %5 : tensor<512x!tt.ptr<f32>>
    tt.return
    }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [512, 256], strides: {{.}}[[CST_256_]], 1] : memref<*xf32> to memref<512x256xf32, strided<[?, 1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<512x256xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<512x256xf32, strided<[?, 1]>> to memref<512x256xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<512x256xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<256x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_transposed_:%.+]] = linalg.transpose ins([[VAR_0_]] : tensor<512x256xf32>) outs([[VAR_1_]] : tensor<256x512xf32>) permutation = [1, 0]
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<512xf32>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_2_]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_transposed_]] : tensor<256x512xf32>) outs([[VAR_3_]] : tensor<512xf32>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: f32, [[init_:%.+]]: f32) {
// CHECK:               [[VAR_4_:%.+]] = arith.addf [[in_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_4_]] : f32
// CHECK:             }
// CHECK:           bufferization.materialize_in_destination [[VAR_reduced_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<512xf32>, memref<512xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
