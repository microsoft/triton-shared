#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%1 : tensor<4x256xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %6 = arith.addf %in, %in_3 : bf16
      linalg.yield %6 : bf16
    } -> tensor<4x256xbf16>
    %5 = arith.index_cast %arg3 : i32 to index
    return
  }
}
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: memref<*xbf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins({{.}}[1_]], {{.}}[3_]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs({{.}}[1_]] : tensor<4x256xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:             linalg.yield [[VAR_6_]] : bf16
// CHECK:           } -> tensor<4x256xbf16>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           return
// CHECK:         }
