// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<f32>,
        %res : !tt.ptr<f32>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : (i32) -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : (tensor<512xi32>) -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : (tensor<512x1xi32>) -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : (!tt.ptr<f32>) -> tensor<512x256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %19 = tt.addptr %18, %3 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %afm = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x256xf32>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<512x256xf32>) -> tensor<256xf32>
    tt.store %19, %5 : tensor<256xf32>
    tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32>, %[[VAL_1:.*]]: memref<*xf32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [512, 256], strides: {{\[}}%[[VAL_5]], 1] : memref<*xf32> to memref<512x256xf32, strided<[?, 1]>>
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<512x256xf32>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_9]] : memref<512x256xf32, strided<[?, 1]>> to memref<512x256xf32>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<512x256xf32>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<256xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_11]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.reduce ins(%[[VAL_10]] : tensor<512x256xf32>) outs(%[[VAL_12]] : tensor<256xf32>) dimensions = [0]
// CHECK:             (%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32) {
// CHECK:               %[[VAL_16:.*]] = arith.addf %[[VAL_14]], %[[VAL_15]] : f32
// CHECK:               linalg.yield %[[VAL_16]] : f32
// CHECK:             }
// CHECK:           bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_8]]
// CHECK:           return
// CHECK:         }
