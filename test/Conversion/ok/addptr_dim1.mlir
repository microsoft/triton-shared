// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>

    %splat_arg0 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<256x1x!tt.ptr<bf16>>
    %2 = tt.addptr %splat_arg0, %1 : tensor<256x1x!tt.ptr<bf16>>, tensor<256x1xi32>

    // 256x1 pointer should have meaningful stride in outer dimension
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x1xbf16>

    %4 = tt.splat %arg1 : (i32) -> tensor<256x1xi32>
    // 256x1 pointer should have meaningful stride in outer dimension
    %5 = tt.addptr %2, %4 : tensor<256x1x!tt.ptr<bf16>>, tensor<256x1xi32>
    tt.store %5, %3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x1x!tt.ptr<bf16>>, tensor<256x1xbf16>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4x256xbf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_0_]] : tensor<4x256xbf16>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<256x1xbf16, strided<[256, 1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<256x1xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<256x1xbf16, strided<[256, 1]>> to memref<256x1xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256x1xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<256x1xbf16, strided<[256, 1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]]
// CHECK-DAG:       [[VAR_4_:%.+]]:3 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg6_:%.+]] = [[VAR_1_]], [[VAR_arg7_:%.+]] = [[CST_0_]], [[VAR_arg8_:%.+]] = [[CST_0_]]) -> (tensor<4x256xbf16>, index, index) {
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.index_cast [[VAR_arg5_]] : index to i32
// CHECK:             [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_256_1_]] : i32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i32 to index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_arg8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_8_]]{{.}}, sizes: [4, 256], strides: {{.}}[[VAR_7_]], [[CST_1_]]{{.}} : memref<*xbf16> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_2_]], [[RES_1_]] : memref<4x256xbf16, strided<[?, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:             [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<4x256xbf16>
// CHECK:             [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg6_]], [[VAR_9_]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs([[VAR_arg6_]] : tensor<4x256xbf16>) {
// CHECK:             ^bb0([[in1:%.+]]: bf16, [[in2:%.+]]: bf16, [[out:%.+]]: bf16):
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[in1]], [[in2]] : bf16
// CHECK:               linalg.yield [[VAR_13_]] : bf16
// CHECK:             } -> tensor<4x256xbf16>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_256_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.addi [[VAR_11_]], [[VAR_arg8_]] : index
// CHECK:             scf.yield [[VAR_10_]], [[VAR_12_]], [[CST_0_]] : tensor<4x256xbf16>, index, index
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [4, 256], strides: {{.}}[[CST_256_]], 1] : memref<*xbf16> to memref<4x256xbf16, strided<[?, 1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]]#0 in writable [[VAR_reinterpret_cast_1_]]
// CHECK:           return
// CHECK:         }
