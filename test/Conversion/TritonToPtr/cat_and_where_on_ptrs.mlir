// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s
// Original triton program:
//    @triton.jit
//    def ptr_cat(in_ptr0, out_ptr0, mask_ptr):
//        offsets = tl.arange(0, 16)
//        ptr_0 = in_ptr0 + tl.arange(0, 8)
//        ptr_1 = out_ptr0 + tl.arange(0, 8)
//        ptr = tl.cat(ptr_0, ptr_1, can_reorder=True)
//        ptr_true = ptr + 4 * tl.load(offsets + ptr)
//        ptr_false = ptr + 5 * tl.load(offsets + ptr)
//        masks = tl.load(mask_ptr + offsets)
//        ptr_load = tl.where(masks, ptr_true, ptr_false)
//        a = tl.load(ptr_load + offsets, mask=masks)
//        tl.store(out_ptr0 + offsets, a, mask=masks)

module {
  tt.func public @ptr_cat(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i1>) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<16xi8>
    %cst_0 = arith.constant dense<5> : tensor<16xi32>
    %cst_1 = arith.constant dense<4> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %3 = tt.addptr %2, %1 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %5 = tt.addptr %4, %1 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %6 = tt.cat %3, %5 : tensor<8x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i32>>
    %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %8 = tt.load %7 : tensor<16x!tt.ptr<i32>>
    %9 = arith.muli %8, %cst_1 : tensor<16xi32>
    %10 = tt.addptr %6, %9 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %11 = arith.muli %8, %cst_0 : tensor<16xi32>
    %12 = tt.addptr %6, %11 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %13 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<16x!tt.ptr<i1>>
    %14 = tt.addptr %13, %0 : tensor<16x!tt.ptr<i1>>, tensor<16xi32>
    %15 = tt.bitcast %14 : tensor<16x!tt.ptr<i1>> -> tensor<16x!tt.ptr<i8>>
    %16 = tt.load %15 : tensor<16x!tt.ptr<i8>>
    %17 = arith.cmpi ne, %16, %cst : tensor<16xi8>
    %18 = arith.select %17, %10, %12 : tensor<16xi1>, tensor<16x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %20 = tt.load %19, %17 : tensor<16x!tt.ptr<i32>>
    %21 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %22 = tt.addptr %21, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %22, %20, %17 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @ptr_cat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: !tt.ptr<i1>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tptr.type_offset i1  : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_1_:%.+]] = tptr.type_offset i32  : i32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : i8
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : !tt.ptr<i1> to !ptr.ptr
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<i32> to !ptr.ptr
// CHECK-DAG:       [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<i32> to !ptr.ptr
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<16xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = linalg.fill ins([[CST_0_2_]] : i8) outs([[VAR_5_]] : tensor<16xi8>) -> tensor<16xi8>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<16xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = linalg.fill ins([[CST_5_]] : i32) outs([[VAR_7_]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = linalg.fill ins([[CST_4_]] : i32) outs([[VAR_7_]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK-DAG:       [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_7_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32):
// CHECK:             [[VAR_35_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_36_:%.+]] = arith.index_cast [[VAR_35_]] : index to i32
// CHECK:             linalg.yield [[VAR_36_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_11_:%.+]] = tensor.empty() : tensor<8xi32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_11_]] : tensor<8xi32>) {
// CHECK:           ^bb0([[IN_1_:%.+]]: i32):
// CHECK:             [[VAR_35_1_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_36_1_:%.+]] = arith.index_cast [[VAR_35_1_]] : index to i32
// CHECK:             linalg.yield [[VAR_36_1_]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           [[VAR_13_:%.+]] = tensor.empty() : tensor<8x!ptr.ptr>
// CHECK:           [[VAR_14_:%.+]] = linalg.fill ins([[VAR_4_]] : !ptr.ptr) outs([[VAR_13_]] : tensor<8x!ptr.ptr>) -> tensor<8x!ptr.ptr>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_14_]], [[VAR_12_]] : tensor<8x!ptr.ptr>, tensor<8xi32>) outs([[VAR_14_]] : tensor<8x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: !ptr.ptr, [[IN_3_:%.+]]: i32, [[IN_4_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_2_:%.+]] = arith.muli [[IN_3_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_2_:%.+]] = tptr.ptradd [[IN_2_]] [[VAR_35_2_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_2_]] : !ptr.ptr
// CHECK:           } -> tensor<8x!ptr.ptr>
// CHECK:           [[VAR_16_:%.+]] = linalg.fill ins([[VAR_3_]] : !ptr.ptr) outs([[VAR_13_]] : tensor<8x!ptr.ptr>) -> tensor<8x!ptr.ptr>
// CHECK:           [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_16_]], [[VAR_12_]] : tensor<8x!ptr.ptr>, tensor<8xi32>) outs([[VAR_16_]] : tensor<8x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_5_:%.+]]: !ptr.ptr, [[IN_6_:%.+]]: i32, [[IN_7_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_3_:%.+]] = arith.muli [[IN_6_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_3_:%.+]] = tptr.ptradd [[IN_5_]] [[VAR_35_3_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_3_]] : !ptr.ptr
// CHECK:           } -> tensor<8x!ptr.ptr>
// CHECK:           [[VAR_18_:%.+]] = tensor.empty() : tensor<16x!ptr.ptr>
// CHECK:           [[VAR_inserted_slice_:%.+]] = tensor.insert_slice [[VAR_15_]] into [[VAR_18_]][0] [8] [1] : tensor<8x!ptr.ptr> into tensor<16x!ptr.ptr>
// CHECK:           [[VAR_inserted_slice_0_:%.+]] = tensor.insert_slice [[VAR_17_]] into [[VAR_inserted_slice_]][8] [8] [1] : tensor<8x!ptr.ptr> into tensor<16x!ptr.ptr>
// CHECK:           [[VAR_19_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_inserted_slice_0_]], [[VAR_10_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_inserted_slice_0_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_8_:%.+]]: !ptr.ptr, [[IN_9_:%.+]]: i32, [[IN_10_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_4_:%.+]] = arith.muli [[IN_9_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_4_:%.+]] = tptr.ptradd [[IN_8_]] [[VAR_35_4_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_4_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_20_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_19_]] : tensor<16x!ptr.ptr>) outs([[VAR_7_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_11_:%.+]]: !ptr.ptr, [[IN_12_:%.+]]: i32):
// CHECK:             [[VAR_35_5_:%.+]] = tptr.to_memref [[IN_11_]] : !ptr.ptr to memref<1xi32>
// CHECK:             [[VAR_36_4_:%.+]] = memref.load [[VAR_35_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             linalg.yield [[VAR_36_4_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_20_]], [[VAR_9_]] : tensor<16xi32>, tensor<16xi32>) outs([[VAR_20_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_13_:%.+]]: i32, [[IN_14_:%.+]]: i32, [[IN_15_:%.+]]: i32):
// CHECK:             [[VAR_35_6_:%.+]] = arith.muli [[IN_13_]], [[IN_14_]] : i32
// CHECK:             linalg.yield [[VAR_35_6_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_22_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_inserted_slice_0_]], [[VAR_21_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_inserted_slice_0_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_16_:%.+]]: !ptr.ptr, [[IN_17_:%.+]]: i32, [[IN_18_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_7_:%.+]] = arith.muli [[IN_17_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_5_:%.+]] = tptr.ptradd [[IN_16_]] [[VAR_35_7_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_5_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_23_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_20_]], [[VAR_8_]] : tensor<16xi32>, tensor<16xi32>) outs([[VAR_20_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_19_:%.+]]: i32, [[IN_20_:%.+]]: i32, [[IN_21_:%.+]]: i32):
// CHECK:             [[VAR_35_8_:%.+]] = arith.muli [[IN_19_]], [[IN_20_]] : i32
// CHECK:             linalg.yield [[VAR_35_8_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_24_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_inserted_slice_0_]], [[VAR_23_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_inserted_slice_0_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_22_:%.+]]: !ptr.ptr, [[IN_23_:%.+]]: i32, [[IN_24_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_9_:%.+]] = arith.muli [[IN_23_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_6_:%.+]] = tptr.ptradd [[IN_22_]] [[VAR_35_9_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_6_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_25_:%.+]] = linalg.fill ins([[VAR_2_]] : !ptr.ptr) outs([[VAR_18_]] : tensor<16x!ptr.ptr>) -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_26_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_25_]], [[VAR_10_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_25_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_25_:%.+]]: !ptr.ptr, [[IN_26_:%.+]]: i32, [[IN_27_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_10_:%.+]] = arith.muli [[IN_26_]], [[VAR_0_]] : i32
// CHECK:             [[VAR_36_7_:%.+]] = tptr.ptradd [[IN_25_]] [[VAR_35_10_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_7_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_27_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_26_]] : tensor<16x!ptr.ptr>) outs([[VAR_5_]] : tensor<16xi8>) {
// CHECK:           ^bb0([[IN_28_:%.+]]: !ptr.ptr, [[IN_29_:%.+]]: i8):
// CHECK:             [[VAR_35_11_:%.+]] = tptr.to_memref [[IN_28_]] : !ptr.ptr to memref<1xi8>
// CHECK:             [[VAR_36_7_:%.+]] = memref.load [[VAR_35_11_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:             linalg.yield [[VAR_36_7_]] : i8
// CHECK:           } -> tensor<16xi8>
// CHECK:           [[VAR_28_:%.+]] = tensor.empty() : tensor<16xi1>
// CHECK:           [[VAR_29_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_27_]], [[VAR_6_]] : tensor<16xi8>, tensor<16xi8>) outs([[VAR_28_]] : tensor<16xi1>) {
// CHECK:           ^bb0([[IN_30_:%.+]]: i8, [[IN_31_:%.+]]: i8, [[IN_32_:%.+]]: i1):
// CHECK:             [[VAR_35_12_:%.+]] = arith.cmpi ne, [[IN_30_]], [[IN_31_]] : i8
// CHECK:             linalg.yield [[VAR_35_12_]] : i1
// CHECK:           } -> tensor<16xi1>
// CHECK:           [[VAR_30_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_29_]], [[VAR_22_]], [[VAR_24_]] : tensor<16xi1>, tensor<16x!ptr.ptr>, tensor<16x!ptr.ptr>) outs([[VAR_22_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_33_:%.+]]: i1, [[IN_34_:%.+]]: !ptr.ptr, [[IN_35_:%.+]]: !ptr.ptr, [[IN_36_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_13_:%.+]] = arith.select [[IN_33_]], [[IN_34_]], [[IN_35_]] : !ptr.ptr
// CHECK:             linalg.yield [[VAR_35_13_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_31_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_30_]], [[VAR_10_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_30_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_37_:%.+]]: !ptr.ptr, [[IN_38_:%.+]]: i32, [[IN_39_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_14_:%.+]] = arith.muli [[IN_38_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_8_:%.+]] = tptr.ptradd [[IN_37_]] [[VAR_35_14_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_8_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_32_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_31_]], [[VAR_29_]] : tensor<16x!ptr.ptr>, tensor<16xi1>) outs([[VAR_7_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[IN_40_:%.+]]: !ptr.ptr, [[IN_41_:%.+]]: i1, [[IN_42_:%.+]]: i32):
// CHECK-DAG:         [[VAR_35_15_:%.+]] = tptr.to_memref [[IN_40_]] : !ptr.ptr to memref<1xi32>
// CHECK-DAG:         [[VAR_36_9_:%.+]] = scf.if [[IN_41_]] -> (i32) {
// CHECK:               [[LOAD_VAR_35_15_MEM_:%.+]] = memref.load [[VAR_35_15_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               scf.yield [[LOAD_VAR_35_15_MEM_]] : i32
// CHECK:             } else {
// CHECK:               scf.yield [[CST_0_]] : i32
// CHECK:             }
// CHECK:             linalg.yield [[VAR_36_9_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           [[VAR_33_:%.+]] = linalg.fill ins([[VAR_3_]] : !ptr.ptr) outs([[VAR_18_]] : tensor<16x!ptr.ptr>) -> tensor<16x!ptr.ptr>
// CHECK:           [[VAR_34_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_33_]], [[VAR_10_]] : tensor<16x!ptr.ptr>, tensor<16xi32>) outs([[VAR_33_]] : tensor<16x!ptr.ptr>) {
// CHECK:           ^bb0([[IN_43_:%.+]]: !ptr.ptr, [[IN_44_:%.+]]: i32, [[IN_45_:%.+]]: !ptr.ptr):
// CHECK:             [[VAR_35_16_:%.+]] = arith.muli [[IN_44_]], [[VAR_1_]] : i32
// CHECK:             [[VAR_36_10_:%.+]] = tptr.ptradd [[IN_43_]] [[VAR_35_16_]] : !ptr.ptr, i32 to !ptr.ptr
// CHECK:             linalg.yield [[VAR_36_10_]] : !ptr.ptr
// CHECK:           } -> tensor<16x!ptr.ptr>
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_34_]], [[VAR_32_]], [[VAR_29_]] : tensor<16x!ptr.ptr>, tensor<16xi32>, tensor<16xi1>) {
// CHECK:           ^bb0([[IN_46_:%.+]]: !ptr.ptr, [[IN_47_:%.+]]: i32, [[IN_48_:%.+]]: i1):
// CHECK:             scf.if [[IN_48_]] {
// CHECK:               [[VAR_35_17_:%.+]] = tptr.to_memref [[IN_46_]] : !ptr.ptr to memref<1xi32>
// CHECK:               memref.store [[IN_47_]], [[VAR_35_17_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
