// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %4, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %2, %9 : i32
    %11 = arith.cmpi slt, %10, %c8_i32 : i32
    %12 = arith.select %11, %10, %c8_i32 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %19 = tt.splat %17 : i32 -> tensor<128xi32>
    %20 = arith.addi %19, %18 : tensor<128xi32>
    %21 = arith.muli %16, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %23 = tt.splat %21 : i32 -> tensor<256xi32>
    %24 = arith.addi %23, %22 : tensor<256xi32>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %26 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %30 = tt.splat %arg7 : i32 -> tensor<1x64xi32>
    %31 = arith.muli %29, %30 : tensor<1x64xi32>
    %32 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.broadcast %31 : tensor<1x64xi32> -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %35 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %37 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %41 = tt.splat %arg9 : i32 -> tensor<1x256xi32>
    %42 = arith.muli %40, %41 : tensor<1x256xi32>
    %43 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x256xi32>
    %44 = tt.broadcast %42 : tensor<1x256xi32> -> tensor<64x256xi32>
    %45 = arith.addi %43, %44 : tensor<64x256xi32>
    %46 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %45 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %48 = tt.splat %cst : f32 -> tensor<128x256xf32>
    %49 = arith.muli %arg7, %c64_i32 : i32
    %50 = tt.splat %49 : i32 -> tensor<128x64xi32>
    %51 = arith.muli %arg8, %c64_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<64x256xi32>
    %53:3 = scf.for %arg12 = %c0_i32 to %6 step %c1_i32 iter_args(%arg13 = %48, %arg14 = %36, %arg15 = %47) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>)  : i32 {
      %71 = tt.load %arg14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xbf16>
      %72 = tt.load %arg15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256xbf16>
      %73 = tt.dot %71, %72, %48 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %74 = arith.addf %arg13, %73 : tensor<128x256xf32>
      %75 = tt.addptr %arg14, %50 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
      %76 = tt.addptr %arg15, %52 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
      scf.yield %74, %75, %76 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>
    }
    %54 = arith.truncf %53#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %55 = tt.splat %arg10 : i32 -> tensor<128x1xi32>
    %56 = arith.muli %55, %26 : tensor<128x1xi32>
    %57 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %59 = tt.splat %arg11 : i32 -> tensor<1x256xi32>
    %60 = arith.muli %59, %40 : tensor<1x256xi32>
    %61 = tt.broadcast %58 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x256x!tt.ptr<bf16>>
    %62 = tt.broadcast %60 : tensor<1x256xi32> -> tensor<128x256xi32>
    %63 = tt.addptr %61, %62 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %64 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %65 = arith.cmpi slt, %26, %64 : tensor<128x1xi32>
    %66 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %67 = arith.cmpi slt, %40, %66 : tensor<1x256xi32>
    %68 = tt.broadcast %65 : tensor<128x1xi1> -> tensor<128x256xi1>
    %69 = tt.broadcast %67 : tensor<1x256xi1> -> tensor<128x256xi1>
    %70 = arith.andi %68, %69 : tensor<128x256xi1>
    tt.store %63, %54, %70 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @matmul_kernel_0123456789101112131415
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: memref<*xbf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32, [[PARAM_15_:%.+]]: i32, [[PARAM_16_:%.+]]: i32, [[PARAM_17_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[CST_128_1_:%.+]] = arith.constant 128 : i32
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_127_:%.+]] = arith.constant 127 : i32
// CHECK-DAG:       [[CST_255_:%.+]] = arith.constant 255 : i32
// CHECK-DAG:       [[CST_63_:%.+]] = arith.constant 63 : i32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<128x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[PARAM_3_]], [[CST_127_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.divsi [[VAR_2_]], [[CST_128_1_]] : i32
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.addi [[PARAM_4_]], [[CST_255_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.divsi [[VAR_4_]], [[CST_256_1_]] : i32
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.addi [[PARAM_5_]], [[CST_63_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.divsi [[VAR_6_]], [[CST_64_]] : i32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.muli [[VAR_5_]], [[CST_8_]] : i32
// CHECK:           [[VAR_9_:%.+]] = arith.divsi [[PARAM_15_]], [[VAR_8_]] : i32
// CHECK:           [[VAR_10_:%.+]] = arith.muli [[VAR_9_]], [[CST_8_]] : i32
// CHECK:           [[VAR_11_:%.+]] = arith.subi [[VAR_3_]], [[VAR_10_]] : i32
// CHECK:           [[VAR_12_:%.+]] = arith.minsi [[VAR_11_]], [[CST_8_]] : i32
// CHECK:           [[VAR_13_:%.+]] = arith.remsi [[PARAM_15_]], [[VAR_12_]] : i32
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.addi [[VAR_10_]], [[VAR_13_]] : i32
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.remsi [[PARAM_15_]], [[VAR_8_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.divsi [[VAR_15_]], [[VAR_12_]] : i32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.muli [[VAR_14_]], [[CST_128_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.muli [[VAR_16_]], [[CST_256_1_]] : i32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.muli [[VAR_19_]], [[VAR_20_]] : index
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.index_cast [[PARAM_8_]] : i32 to index
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.index_cast [[VAR_18_]] : i32 to index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.index_cast [[PARAM_9_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.muli [[VAR_24_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.muli [[PARAM_7_]], [[CST_64_]] : i32
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.muli [[PARAM_8_]], [[CST_64_]] : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_21_]]{{.}}, sizes: [128, 64], strides: {{.}}[[VAR_20_]], [[VAR_22_]]{{.}} : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_26_]]{{.}}, sizes: [64, 256], strides: {{.}}[[VAR_23_]], [[VAR_25_]]{{.}} : memref<*xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_29_:%.+]]:7 = scf.for [[VAR_arg18_:%.+]] = [[CST_0_1_]] to [[VAR_7_]] step [[CST_1_]] iter_args([[VAR_arg19_:%.+]] = [[VAR_1_]], [[VAR_arg20_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg21_:%.+]] = [[VAR_reinterpret_cast_]]_0, [[VAR_arg22_:%.+]] = [[VAR_21_]], [[VAR_arg23_:%.+]] = [[CST_0_]], [[VAR_arg24_:%.+]] = [[VAR_26_]], [[VAR_arg25_:%.+]] = [[CST_0_]]) -> (tensor<128x256xf32>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<64x256xbf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<128x64xbf16>
// CHECK:             memref.copy [[VAR_arg20_]], [[RES_]] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
// CHECK-DAG:         [[VAR_51_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x64xbf16>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<64x256xbf16>
// CHECK:             memref.copy [[VAR_arg21_]], [[RES_1_]] : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<64x256xbf16>
// CHECK-DAG:         [[VAR_52_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<64x256xbf16>
// CHECK-DAG:         [[VAR_53_:%.+]] = tensor.empty() : tensor<128x256xf32>
// CHECK:             [[VAR_54_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_53_]] : tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:             [[VAR_55_:%.+]] = linalg.matmul ins([[VAR_51_]], [[VAR_52_]] : tensor<128x64xbf16>, tensor<64x256xbf16>) outs([[VAR_54_]] : tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:             [[VAR_56_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_55_]], [[VAR_1_]] : tensor<128x256xf32>, tensor<128x256xf32>) outs([[VAR_55_]] : tensor<128x256xf32>) {
// CHECK:             ^bb0([[in_:.+]]: f32, [[in_1:.+]]: f32, [[out_:.+]]: f32):
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[in_]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_64_]] : f32
// CHECK:             } -> tensor<128x256xf32>
// CHECK:             [[VAR_57_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg19_]], [[VAR_56_]] : tensor<128x256xf32>, tensor<128x256xf32>) outs([[VAR_arg19_]] : tensor<128x256xf32>) {
// CHECK:             ^bb0([[in_]]: f32, [[in_1:.+]]: f32, [[out_:.+]]: f32):
// CHECK:               [[VAR_64_1_:%.+]] = arith.addf [[in_]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_64_1_]] : f32
// CHECK:             } -> tensor<128x256xf32>
// CHECK:             [[VAR_58_:%.+]] = arith.index_cast [[VAR_27_]] : i32 to index
// CHECK:             [[VAR_59_:%.+]] = arith.addi [[VAR_arg22_]], [[VAR_58_]] : index
// CHECK:             [[VAR_60_:%.+]] = arith.addi [[VAR_59_]], [[VAR_arg23_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_60_]]{{.}}, sizes: [128, 64], strides: {{.}}[[VAR_20_]], [[VAR_22_]]{{.}} : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.index_cast [[VAR_28_]] : i32 to index
// CHECK:             [[VAR_62_:%.+]] = arith.addi [[VAR_arg24_]], [[VAR_61_]] : index
// CHECK:             [[VAR_63_:%.+]] = arith.addi [[VAR_62_]], [[VAR_arg25_]] : index
// CHECK:             [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_63_]]{{.}}, sizes: [64, 256], strides: {{.}}[[VAR_23_]], [[VAR_25_]]{{.}} : memref<*xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
// CHECK:             scf.yield [[VAR_57_]], [[VAR_reinterpret_cast_3_]], [[VAR_reinterpret_cast_4_]], [[VAR_60_]], [[CST_0_]], [[VAR_63_]], [[CST_0_]] : tensor<128x256xf32>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<64x256xbf16, strided<[?, ?], offset: ?>>, index, index, index, index
// CHECK:           }
// CHECK:           [[VAR_30_:%.+]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           [[VAR_31_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_29_]]#0 : tensor<128x256xf32>) outs([[VAR_30_]] : tensor<128x256xbf16>) {
// CHECK:           ^bb0([[in_:.+]]: f32, [[out_:.+]]: bf16):
// CHECK:             [[VAR_51_1_:%.+]] = arith.truncf [[in_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_51_1_]] : bf16
// CHECK:           } -> tensor<128x256xbf16>
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.index_cast [[PARAM_10_]] : i32 to index
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.muli [[VAR_33_]], [[VAR_32_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.index_cast [[PARAM_11_]] : i32 to index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.index_cast [[VAR_18_]] : i32 to index
// CHECK:           [[VAR_37_:%.+]] = arith.muli [[VAR_36_]], [[VAR_35_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.addi [[VAR_34_]], [[VAR_37_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_38_]]{{.}}, sizes: [128, 256], strides: {{.}}[[VAR_32_]], [[VAR_35_]]{{.}} : memref<*xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_39_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_40_:%.+]] = arith.addi [[VAR_39_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_41_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_42_:%.+]] = arith.minsi [[VAR_40_]], [[VAR_41_]] : index
// CHECK-DAG:       [[VAR_43_:%.+]] = arith.subi [[VAR_42_]], [[VAR_39_]] : index
// CHECK-DAG:       [[VAR_44_:%.+]] = arith.index_cast [[VAR_18_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = arith.addi [[VAR_44_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_46_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_47_:%.+]] = arith.minsi [[VAR_45_]], [[VAR_46_]] : index
// CHECK-DAG:       [[VAR_48_:%.+]] = arith.subi [[VAR_47_]], [[VAR_44_]] : index
// CHECK-DAG:       [[VAR_49_:%.+]] = arith.minsi [[VAR_43_]], [[CST_128_]] : index
// CHECK:           [[VAR_50_:%.+]] = arith.minsi [[VAR_48_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_31_]][0, 0] {{.}}[[VAR_49_]], [[VAR_50_]]{{.}} [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0, 0] {{.}}[[VAR_49_]], [[VAR_50_]]{{.}} [1, 1] : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_]]
// CHECK:           return
// CHECK:         }
