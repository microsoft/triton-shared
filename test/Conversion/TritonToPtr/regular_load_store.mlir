// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg"  --triton-to-ptr --canonicalize --cse %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>
    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tptr.type_offset f32  : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tptr.type_offset i32  : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<f32> to !ptr.ptr<#ptr.generic_space>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_4_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_14_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : index to i32
// CHECK:             linalg.yield [[VAR_15_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<1024x!ptr.ptr<#ptr.generic_space>>
// CHECK:           [[VAR_7_:%.+]] = linalg.fill ins([[VAR_3_]] : !ptr.ptr<#ptr.generic_space>) outs([[VAR_6_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>) -> tensor<1024x!ptr.ptr<#ptr.generic_space>>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_5_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>, tensor<1024xi32>) outs([[VAR_7_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0([[IN_1_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             [[VAR_14_1_:%.+]] = arith.muli [[IN_2_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_15_1_:%.+]] = tptr.ptradd [[IN_1_]] [[VAR_14_1_]] : <#ptr.generic_space>, i32 to <#ptr.generic_space>
// CHECK:             linalg.yield [[VAR_15_1_]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<1024x!ptr.ptr<#ptr.generic_space>>
// CHECK:           [[VAR_9_:%.+]] = linalg.fill ins([[VAR_2_]] : !ptr.ptr<#ptr.generic_space>) outs([[VAR_6_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>) -> tensor<1024x!ptr.ptr<#ptr.generic_space>>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]], [[VAR_5_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>, tensor<1024xi32>) outs([[VAR_9_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[IN_5_:%.+]]: i32, [[IN_6_:%.+]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             [[VAR_14_2_:%.+]] = arith.muli [[IN_5_]], [[VAR_0_]] : i32
// CHECK:             [[VAR_15_2_:%.+]] = tptr.ptradd [[IN_4_]] [[VAR_14_2_]] : <#ptr.generic_space>, i32 to <#ptr.generic_space>
// CHECK:             linalg.yield [[VAR_15_2_]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<1024x!ptr.ptr<#ptr.generic_space>>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_8_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>) outs([[VAR_4_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[IN_7_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[IN_8_:%.+]]: i32):
// CHECK:             [[VAR_14_3_:%.+]] = tptr.to_memref [[IN_7_]] : <#ptr.generic_space> to memref<1xi32>
// CHECK:             [[VAR_15_2_:%.+]] = memref.load [[VAR_14_3_]]{{.}}[[CST_0_]]{{.}} : memref<1xi32>
// CHECK:             linalg.yield [[VAR_15_2_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_12_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_13_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_11_]] : tensor<1024xi32>) outs([[VAR_12_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_9_:%.+]]: i32, [[IN_10_:%.+]]: f32):
// CHECK:             [[VAR_14_4_:%.+]] = arith.bitcast [[IN_9_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_14_4_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_10_]], [[VAR_13_]] : tensor<1024x!ptr.ptr<#ptr.generic_space>>, tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_11_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[IN_12_:%.+]]: f32):
// CHECK:             [[VAR_14_5_:%.+]] = tptr.to_memref [[IN_11_]] : <#ptr.generic_space> to memref<1xf32>
// CHECK:             memref.store [[IN_12_]], [[VAR_14_5_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
