// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
  // offset = 0, size = 4, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
  // offset = [0,0], size = [4,1], stride = [1,0]
  %2 = tt.broadcast %1 : (tensor<4x1xi32>) -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [1,0]
  %arg2splat = tt.splat %arg2 : (i32) -> tensor<4x256xi32>
  %offset2 = arith.addi %2, %arg2splat : tensor<4x256xi32>
  // offset = [%arg2,0], size = [4,256], stride = [1,0]
  %arg3splat = tt.splat %arg3 : (i32) -> tensor<4x256xi32>
  %offset3 = arith.addi %offset2, %arg3splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3,0], size = [4,256], stride = [1,0]
  %c10 = arith.constant 10 : i32
  %c10splat = tt.splat %c10 : (i32) -> tensor<4x256xi32>
  %offset4 = arith.addi %offset3, %c10splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3+10,0], size = [4,256], stride = [1,0]
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
  // offset = 0, size = 256, stride = 1
  %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
  // offset = [0,0], size = [1,256], stride = [0,1]
  %5 = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : (i32) -> tensor<4x256xi32>
  %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,6]
  %7 = arith.addi %offset4, %scale5: tensor<4x256xi32>
  // offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %8 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>,tensor<4x256xi32>
  // source = %arg0, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %10 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source = %arg1, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %12 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x256xbf16>
  tt.store %11, %12 : tensor<4x256xbf16>
  tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_1_:%.+]]: !tt.ptr<bf16, 1>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32) {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[CST_10_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [1, [[CST_6_]]{{.}}, offsets: {{.}}[[VAR_3_]], 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[VAR_6_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[CST_10_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4, 256], strides: [1, [[CST_6_]]{{.}}, offsets: {{.}}[[VAR_8_]], 0], shape: [0, 0], order: [] : <bf16, 1> to tensor<4x256x!tt.ptr<bf16, 1>>
// CHECK-DAG:       [[VAR_10_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>) -> tensor<4x256xbf16>
// CHECK:           "tts.store"([[VAR_9_]], [[VAR_10_]]) <{static_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16, 1>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
