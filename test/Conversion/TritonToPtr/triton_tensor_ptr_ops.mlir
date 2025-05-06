// Test all triton ops on tensor of pointers are converted to linalg.generic
// and all triton ops on pointers are converted to ops in the pointer dialect.
// Original triton program:
//    @triton.jit
//    def tensor_ptr(in_ptr0, out_ptr0):
//        ints = tl.load(in_ptr0 + tl.arange(0, 16)).to(tl.int64)
//        ptrs = ints.to(tl.pointer_type(tl.int32))
//        vals = tl.load(ptrs)
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        ints_2 = out_ptrs.to(tl.int64) + vals
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        out_ptrs_i64 = out_ptrs.to(tl.pointer_type(tl.int64))
//        out_ptrs_i64 += 2
//        out_ptrs_i32 = out_ptrs_i64.to(tl.pointer_type(tl.int32))
//        tl.store(out_ptrs_i32, ints_2.to(tl.int32))

// RUN: triton-shared-opt  --triton-arith-to-linalg="tensor-ptr-to-linalg"  --triton-to-ptr --canonicalize --cse %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>>
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %9 = tt.ptr_to_int %8 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %10 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64>
    %11 = arith.addi %9, %10 : tensor<16xi64>
    %12 = tt.bitcast %8 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %13 = tt.addptr %12, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %14 = tt.bitcast %13 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %15 = arith.trunci %11 : tensor<16xi64> to tensor<16xi32>
    tt.store %14, %15 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @tensor_ptr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tptr.type_offset i64  : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_1_:%.+]] = tptr.type_offset i32  : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<16xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = linalg.fill ins([[CST_2_]] : i32) outs([[VAR_4_]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_4_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_22_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_7_:%.+]] = tensor.empty() : tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[VAR_3_]] : !ptr.ptr<#tptr.default_memory_space>) outs([[VAR_7_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_8_]], [[VAR_6_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs([[VAR_8_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0([[IN_1_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             [[VAR_22_1_:%.+]] = arith.muli [[IN_2_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_23_1_:%.+]] = tptr.ptradd [[IN_1_]] [[VAR_22_1_]] : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
// CHECK:             linalg.yield [[VAR_23_1_]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs([[VAR_4_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_5_:%.+]]: i32):
// CHECK:             [[VAR_22_2_:%.+]] = tptr.to_memref [[IN_4_]] : <#tptr.default_memory_space> to memref<1xi32>
// CHECK:             [[VAR_23_1_:%.+]] = memref.load [[VAR_22_2_]]{{.}}[[CST_0_]]{{.}} : memref<1xi32>
// CHECK:             linalg.yield [[VAR_23_1_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_11_:%.+]] = tensor.empty() : tensor<16xi64>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_10_]] : tensor<16xi32>) outs([[VAR_11_]] : tensor<16xi64>) {
// CHECK:           ^bb0([[IN_6_:%.+]]: i32, [[IN_7_:%.+]]: i64):
// CHECK:             [[VAR_22_3_:%.+]] = arith.extsi [[IN_6_]] : i32 to i64
// CHECK:             linalg.yield [[VAR_22_3_]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           [[VAR_13_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_12_]] : tensor<16xi64>) outs([[VAR_7_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0([[IN_8_:%.+]]: i64, [[IN_9_:%.+]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             [[VAR_22_4_:%.+]] = tptr.inttoptr [[IN_8_]] : i64 to <#tptr.default_memory_space>
// CHECK:             linalg.yield [[VAR_22_4_]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_13_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs([[VAR_4_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_10_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_11_:%.+]]: i32):
// CHECK:             [[VAR_22_5_:%.+]] = tptr.to_memref [[IN_10_]] : <#tptr.default_memory_space> to memref<1xi32>
// CHECK:             [[VAR_23_1_1_:%.+]] = memref.load [[VAR_22_5_]]{{.}}[[CST_0_]]{{.}} : memref<1xi32>
// CHECK:             linalg.yield [[VAR_23_1_1_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_15_:%.+]] = linalg.fill ins([[VAR_2_]] : !ptr.ptr<#tptr.default_memory_space>) outs([[VAR_7_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_15_]], [[VAR_6_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs([[VAR_15_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0([[IN_12_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_13_:%.+]]: i32, [[IN_14_:%.+]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             [[VAR_22_6_:%.+]] = arith.muli [[IN_13_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_23_2_:%.+]] = tptr.ptradd [[IN_12_]] [[VAR_22_6_]] : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
// CHECK:             linalg.yield [[VAR_23_2_]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_16_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs([[VAR_11_]] : tensor<16xi64>) {
// CHECK:           ^bb0([[IN_15_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_16_:%.+]]: i64):
// CHECK:             [[VAR_22_7_:%.+]] = tptr.ptrtoint [[IN_15_]] : <#tptr.default_memory_space> to i64
// CHECK:             linalg.yield [[VAR_22_7_]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_14_]] : tensor<16xi32>) outs([[VAR_11_]] : tensor<16xi64>) {
// CHECK:           ^bb0([[IN_17_:%.+]]: i32, [[IN_18_:%.+]]: i64):
// CHECK:             [[VAR_22_8_:%.+]] = arith.extsi [[IN_17_]] : i32 to i64
// CHECK:             linalg.yield [[VAR_22_8_]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           [[VAR_19_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_17_]], [[VAR_18_]] : tensor<16xi64>, tensor<16xi64>) outs([[VAR_17_]] : tensor<16xi64>) {
// CHECK:           ^bb0([[IN_19_:%.+]]: i64, [[IN_20_:%.+]]: i64, [[IN_21_:%.+]]: i64):
// CHECK:             [[VAR_22_9_:%.+]] = arith.addi [[IN_19_]], [[IN_20_]] : i64
// CHECK:             linalg.yield [[VAR_22_9_]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           [[VAR_20_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_16_]], [[VAR_5_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs([[VAR_16_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0([[IN_22_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_23_:%.+]]: i32, [[IN_24_:%.+]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             [[VAR_22_10_:%.+]] = arith.muli [[IN_23_]], [[VAR_0_]] : i32
// CHECK:             [[VAR_23_3_:%.+]] = tptr.ptradd [[IN_22_]] [[VAR_22_10_]] : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
// CHECK:             linalg.yield [[VAR_23_3_]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_19_]] : tensor<16xi64>) outs([[VAR_4_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_25_:%.+]]: i64, [[IN_26_:%.+]]: i32):
// CHECK:             [[VAR_22_11_:%.+]] = arith.trunci [[IN_25_]] : i64 to i32
// CHECK:             linalg.yield [[VAR_22_11_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_20_]], [[VAR_21_]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) {
// CHECK:           ^bb0([[IN_27_:%.+]]: !ptr.ptr<#tptr.default_memory_space>, [[IN_28_:%.+]]: i32):
// CHECK:             [[VAR_22_12_:%.+]] = tptr.to_memref [[IN_27_]] : <#tptr.default_memory_space> to memref<1xi32>
// CHECK:             memref.store [[IN_28_]], [[VAR_22_12_]]{{.}}[[CST_0_]]{{.}} : memref<1xi32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
