// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @_layer_norm_fwd_fused_0123456789(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32, %arg7: i32, %arg8: f32) {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %cst_0 : f32 -> tensor<256xf32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %7 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %5 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %6 : tensor<256xi32>
      %35 = tt.addptr %7, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.addf %arg10, %36 : tensor<256xf32>
      scf.yield %37 : tensor<256xf32>
    }
    %9 = "tt.reduce"(%8) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %10 = arith.sitofp %arg7 : i32 to f32
    %11 = arith.divf %9, %10 : f32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %13 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %14 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.splat %11 : f32 -> tensor<256xf32>
    %16 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %12 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %13 : tensor<256xi32>
      %35 = tt.addptr %14, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.subf %36, %15 : tensor<256xf32>
      %38 = arith.select %34, %37, %4 : tensor<256xi1>, tensor<256xf32>
      %39 = arith.mulf %38, %38 : tensor<256xf32>
      %40 = arith.addf %arg10, %39 : tensor<256xf32>
      scf.yield %40 : tensor<256xf32>
    }
    %17 = "tt.reduce"(%16) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %18 = arith.divf %17, %10 : f32
    %19 = arith.addf %18, %arg8 : f32
    %20 = math.sqrt %19 : f32
    %21 = arith.divf %cst, %20 : f32
    %22 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %22, %11 : !tt.ptr<f32>
    %23 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %23, %21 : !tt.ptr<f32>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %25 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %27 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %28 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %29 = tt.splat %11 : f32 -> tensor<256xf32>
    %30 = tt.splat %21 : f32 -> tensor<256xf32>
    %31 = tt.splat %2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %24 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %25 : tensor<256xi32>
      %35 = tt.addptr %26, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34 : tensor<256x!tt.ptr<f32>>
      %37 = tt.addptr %27, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %38 = tt.load %37, %34 : tensor<256x!tt.ptr<f32>>
      %39 = tt.addptr %28, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %40 = tt.load %39, %34, %4 : tensor<256x!tt.ptr<f32>>
      %41 = arith.subf %40, %29 : tensor<256xf32>
      %42 = arith.mulf %41, %30 : tensor<256xf32>
      %43 = arith.mulf %42, %36 : tensor<256xf32>
      %44 = arith.addf %43, %38 : tensor<256xf32>
      %45 = tt.addptr %31, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %45, %44, %34 : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @_layer_norm_fwd_fused_0123456789
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: memref<*xf32>, [[PARAM_5_:%.+]]: memref<*xf32>, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: f32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32

// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_12_]], [[PARAM_6_]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index

// CHECK-DAG:       [[VAR_4_:%.+]] = scf.for [[IV_0_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_]] iter_args([[VAR_arg_0_:%.+]] = [[VAR_1_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.index_cast [[IV_0_]] : i32 to index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.addi [[VAR_3_]], [[VAR_20_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_21_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.addi [[VAR_20_]], [[CST_256_1_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.minsi [[VAR_22_]], [[VAR_23_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.maxsi [[VAR_24_]], [[VAR_20_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.subi [[VAR_25_]], [[VAR_20_]] : index
// CHECK-DAG:         [[ALLOC_0_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[CMP_0_:%.+]] = arith.cmpi slt, [[VAR_26_]], [[CST_256_1_]] : index
// CHECK:             scf.if [[CMP_0_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[ALLOC_0_]] : memref<256xf32>)
// CHECK:             }
// CHECK-DAG:         [[VAR_subview_0_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_]][0] {{.}}[[VAR_26_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_0_:%.+]] = memref.subview [[ALLOC_0_]][0] {{.}}[[VAR_26_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:             memref.copy [[VAR_subview_0_]], [[VAR_subview_6_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_28_:%.+]] = bufferization.to_tensor [[ALLOC_0_]] restrict writable : memref<256xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg_0_]], [[VAR_28_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_arg_0_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[OUT_0_:%.+]]: f32):
// CHECK:               [[ADD_0_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : f32
// CHECK:               linalg.yield [[ADD_0_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             scf.yield [[VAR_29_]] : tensor<256xf32>
// CHECK:           }

// CHECK-DAG:       [[ALLOC_TENSOR_1_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK-DAG:       [[INSERTED_1_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[ALLOC_TENSOR_1_]][] : tensor<f32>
// CHECK-DAG:       [[REDUCED_0_:%.+]] = linalg.reduce ins([[VAR_4_]] : tensor<256xf32>) outs([[INSERTED_1_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[IN_2_:%.+]]: f32, [[INIT_0_:%.+]]: f32) {
// CHECK:               [[ADD_1_:%.+]] = arith.addf [[IN_2_]], [[INIT_0_]] : f32
// CHECK:               linalg.yield [[ADD_1_]] : f32
// CHECK:             }
// CHECK-DAG:       [[EXTRACTED_0_:%.+]] = tensor.extract [[REDUCED_0_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.divf [[EXTRACTED_0_]], [[VAR_6_]] : f32

// CHECK-DAG:       [[VAR_8_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_8_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[OUT_1_:%.+]]: i32):
// CHECK:             [[IDX_0_:%.+]] = linalg.index 0 : index
// CHECK:             [[IDX2I32_0_:%.+]] = arith.index_cast [[IDX_0_]] : index to i32
// CHECK:             linalg.yield [[IDX2I32_0_]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK-DAG:       [[VAR_10_:%.+]] = linalg.fill ins([[PARAM_7_]] : i32) outs([[VAR_8_]] : tensor<256xi32>) -> tensor<256xi32>

// CHECK-DAG:       [[VAR_11_:%.+]] = linalg.fill ins([[VAR_7_]] : f32) outs([[VAR_0_]] : tensor<256xf32>) -> tensor<256xf32>

// CHECK-DAG:       [[VAR_12_:%.+]] = scf.for [[IV_1_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_]] iter_args([[ARG_16_1_:%.+]] = [[VAR_1_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_20_1_:%.+]] = linalg.fill ins([[IV_1_]] : i32) outs([[VAR_8_]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK-DAG:         [[VAR_21_1_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_20_1_]], [[VAR_9_]] : tensor<256xi32>, tensor<256xi32>) outs([[VAR_20_1_]] : tensor<256xi32>) {
// CHECK:             ^bb0([[IN_3_:%.+]]: i32, [[IN_4_:%.+]]: i32, [[OUT_2_:%.+]]: i32):
// CHECK:               [[ADD_2_:%.+]] = arith.addi [[IN_3_]], [[IN_4_]] : i32
// CHECK:               linalg.yield [[ADD_2_]] : i32
// CHECK:             } -> tensor<256xi32>
// CHECK-DAG:         [[TENSOR_I1_:%.+]] = tensor.empty() : tensor<256xi1>
// CHECK-DAG:         [[VAR_23_1_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_21_1_]], [[VAR_10_]] : tensor<256xi32>, tensor<256xi32>) outs([[TENSOR_I1_]] : tensor<256xi1>) {
// CHECK:             ^bb0([[IN_5_:%.+]]: i32, [[IN_6_:%.+]]: i32, [[OUT_3_:%.+]]: i1):
// CHECK:               [[CMP_1_:%.+]] = arith.cmpi slt, [[IN_5_]], [[IN_6_]] : i32
// CHECK:               linalg.yield [[CMP_1_]] : i1
// CHECK:             } -> tensor<256xi1>
// CHECK-DAG:         [[IDX_1_:%.+]] = arith.index_cast [[IV_1_]] : i32 to index
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.addi [[VAR_3_]], [[IDX_1_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_25_1_]]{{.}}, sizes: [256], strides: [1]

// ... (Check the subview, fill, copy, to_tensor, and the subsequent linalg.generic ops) ...

// CHECK:           scf.yield [[ARG_16_2_:%.+]] : tensor<256xf32>
// CHECK:         }

// CHECK-DAG:       [[ALLOC_TENSOR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK-DAG:       [[INSERTED_2_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[ALLOC_TENSOR_2_]][] : tensor<f32>
// CHECK-DAG:       [[REDUCED_2_:%.+]] = linalg.reduce ins([[VAR_12_]] : tensor<256xf32>) outs([[INSERTED_2_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[IN_7_:%.+]]: f32, [[INIT_1_:%.+]]: f32) {
// CHECK:               [[ADD_3_:%.+]] = arith.addf [[IN_7_]], [[INIT_1_]] : f32
// CHECK:               linalg.yield [[ADD_3_]] : f32
// CHECK:             }
// CHECK-DAG:       [[EXTRACTED_3_:%.+]] = tensor.extract [[REDUCED_2_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.divf [[EXTRACTED_3_]], [[VAR_6_]] : f32
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.addf [[VAR_14_]], [[PARAM_8_]] : f32
// CHECK-DAG:       [[VAR_16_:%.+]] = math.sqrt [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[PARAM_12_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_4_]] to offset: {{.}}[[VAR_18_]]{{.}}, sizes: [1], strides: [1]
// CHECK:           affine.store [[VAR_7_]], [[VAR_reinterpret_cast_0_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_5_]] to offset: {{.}}[[VAR_18_]]{{.}}, sizes: [1], strides: [1]
// CHECK:           affine.store [[VAR_17_]], [[VAR_reinterpret_cast_4_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_19_:%.+]] = linalg.fill ins([[VAR_17_]] : f32) outs([[VAR_0_]] : tensor<256xf32>) -> tensor<256xf32>

// CHECK-DAG:       scf.for [[IV_2_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_]]  : i32 {
// CHECK-DAG:         [[VAR_20_2_:%.+]] = arith.index_cast [[IV_2_]] : i32 to index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_2_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_20_2_]]{{.}}, sizes: [256], strides: [1]
// CHECK-DAG:         [[VAR_21_2_:%.+]] = arith.addi [[VAR_20_2_]], [[CST_256_1_]] : index
// CHECK-DAG:         [[VAR_22_2_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:         [[VAR_23_2_:%.+]] = arith.minsi [[VAR_21_2_]], [[VAR_22_2_]] : index
// CHECK-DAG:         [[VAR_24_2_:%.+]] = arith.maxsi [[VAR_23_2_]], [[VAR_20_2_]] : index
// CHECK-DAG:         [[VAR_25_2_:%.+]] = arith.subi [[VAR_24_2_]], [[VAR_20_2_]] : index
// CHECK-DAG:         [[ALLOC_1_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[SUBVIEW_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_2_]][0] {{.}}[[VAR_25_2_]]{{.}} [1]

// ... (Check the subsequent subviews, copies, to_tensor, linalg.generic) ...

// CHECK-DAG:         [[VAR_reinterpret_cast_7_:.+]] = memref.reinterpret_cast [[PARAM_3_]] to offset: {{.}}[[VAR_20_2_]]{{.}}, sizes: [256], strides: [1]
// CHECK-DAG:         [[ALLOC_2_:%.+]] = memref.alloc() : memref<256xf32>

// ... (Fill, copy, and so on) ...

// CHECK-DAG:         [[VAR_34_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_33_:%.+]], [[VAR_27_:%.+]]) outs([[VAR_33_:.+]]) {
// CHECK:             ^bb0([[IN_20_:%.+]]: f32, [[IN_21_:%.+]]: f32, [[OUT_4_:%.+]]: f32):
// CHECK:               [[ADD_5_:%.+]] = arith.addf [[IN_20_]], [[IN_21_]] : f32
// CHECK:               linalg.yield [[ADD_5_]] : f32
// CHECK:             } -> tensor<256xf32>

// CHECK-DAG:         [[VAR_reinterpret_cast_15_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [[SOME_INDEX:.+]], sizes: [256], strides: [1]
// CHECK-DAG:         [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_31_:%.+]][0] {{.}}[[VAR_25_2_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK-DAG:         [[VAR_subview_16_:%.+]] = memref.subview [[VAR_reinterpret_cast_15_]][0] {{.}}[[VAR_25_2_]]{{.}} [1]
// CHECK:             bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_16_]] : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
