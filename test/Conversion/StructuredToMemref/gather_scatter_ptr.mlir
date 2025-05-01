// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure tts.make_indirect_tptr is generated with correct indirect_dim and indirect_offset.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @row_gather2(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_22:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<16x1xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_20]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_25:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_26:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_25]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_27:.*]]: i32):
// CHECK:             %[[VAL_28:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_29:.*]] = arith.index_cast %[[VAL_28]] : index to i32
// CHECK:             linalg.yield %[[VAL_29]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_30:.*]] = tensor.expand_shape %[[VAL_26]] {{\[\[}}0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
// CHECK:           %[[VAL_31:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_30]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_30]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i32):
// CHECK:             %[[VAL_35:.*]] = arith.remsi %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:             linalg.yield %[[VAL_35]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_36:.*]] = arith.muli %[[VAL_3]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_37:.*]] = linalg.fill ins(%[[VAL_36]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_38:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_37]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_31]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_39:.*]]: i32, %[[VAL_40:.*]]: i32, %[[VAL_41:.*]]: i32):
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_39]], %[[VAL_40]] : i32
// CHECK:             linalg.yield %[[VAL_42]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_43:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_30]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_30]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: i32):
// CHECK:             %[[VAL_47:.*]] = arith.divsi %[[VAL_44]], %[[VAL_45]] : i32
// CHECK:             linalg.yield %[[VAL_47]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_48:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_49:.*]] = arith.muli %[[VAL_48]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_50:.*]] = arith.index_cast %[[VAL_49]] : i64 to index
// CHECK:           %[[VAL_51:.*]] = tensor.empty() : tensor<16x1xi64>
// CHECK:           %[[VAL_52:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_38]] : tensor<16x1xi32>) outs(%[[VAL_51]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_53:.*]]: i32, %[[VAL_54:.*]]: i64):
// CHECK:             %[[VAL_55:.*]] = arith.extsi %[[VAL_53]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_55]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_56:.*]] = linalg.fill ins(%[[VAL_8]] : i64) outs(%[[VAL_51]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_57:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_52]], %[[VAL_56]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_52]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_58:.*]]: i64, %[[VAL_59:.*]]: i64, %[[VAL_60:.*]]: i64):
// CHECK:             %[[VAL_61:.*]] = arith.muli %[[VAL_58]], %[[VAL_59]] : i64
// CHECK:             linalg.yield %[[VAL_61]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_62:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_63:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_43]], %[[VAL_62]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_43]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_64:.*]]: i32, %[[VAL_65:.*]]: i32, %[[VAL_66:.*]]: i32):
// CHECK:             %[[VAL_67:.*]] = arith.addi %[[VAL_64]], %[[VAL_65]] : i32
// CHECK:             linalg.yield %[[VAL_67]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_68:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_69:.*]] = arith.index_cast %[[VAL_68]] : index to i32
// CHECK:           %[[VAL_70:.*]] = linalg.fill ins(%[[VAL_69]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_71:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_63]], %[[VAL_70]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_63]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_72:.*]]: i32, %[[VAL_73:.*]]: i32, %[[VAL_74:.*]]: i32):
// CHECK:             %[[VAL_75:.*]] = arith.muli %[[VAL_72]], %[[VAL_73]] : i32
// CHECK:             linalg.yield %[[VAL_75]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_76:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_57]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_77:.*]]: i64, %[[VAL_78:.*]]: i32):
// CHECK:             %[[VAL_79:.*]] = arith.trunci %[[VAL_77]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_79]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_80:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_71]], %[[VAL_76]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_71]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_81:.*]]: i32, %[[VAL_82:.*]]: i32, %[[VAL_83:.*]]: i32):
// CHECK:             %[[VAL_84:.*]] = arith.addi %[[VAL_81]], %[[VAL_82]] : i32
// CHECK:             linalg.yield %[[VAL_84]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_85:.*]] = arith.index_cast %[[VAL_50]] : index to i32
// CHECK:           %[[VAL_86:.*]] = linalg.fill ins(%[[VAL_85]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_87:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_86]], %[[VAL_80]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_86]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_88:.*]]: i32, %[[VAL_89:.*]]: i32, %[[VAL_90:.*]]: i32):
// CHECK:             %[[VAL_91:.*]] = arith.addi %[[VAL_88]], %[[VAL_89]] : i32
// CHECK:             linalg.yield %[[VAL_91]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_92:.*]] = tensor.collapse_shape %[[VAL_87]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_93:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           scf.for %[[VAL_94:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_95:.*]] = tensor.extract %[[VAL_92]]{{\[}}%[[VAL_94]]] : tensor<16xi32>
// CHECK:             %[[VAL_96:.*]] = arith.index_cast %[[VAL_95]] : i32 to index
// CHECK:             %[[VAL_97:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_96]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_98:.*]] = memref.subview %[[VAL_97]][0, 0] [1, 8] [16, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             %[[VAL_99:.*]] = memref.subview %[[VAL_93]]{{\[}}%[[VAL_94]], 0] [1, 8] [16, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[256, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_98]], %[[VAL_99]] : memref<1x8xf32, strided<[16, 1], offset: ?>> to memref<1x8xf32, strided<[256, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_100:.*]] = bufferization.to_tensor %[[VAL_93]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_101:.*]] = arith.muli %[[VAL_48]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_102:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_103:.*]] = arith.muli %[[VAL_102]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_104:.*]] = arith.addi %[[VAL_101]], %[[VAL_103]] : i64
// CHECK:           %[[VAL_105:.*]] = arith.index_cast %[[VAL_104]] : i64 to index
// CHECK:           %[[VAL_106:.*]] = linalg.fill ins(%[[VAL_11]] : i64) outs(%[[VAL_51]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_107:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_52]], %[[VAL_106]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_52]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_108:.*]]: i64, %[[VAL_109:.*]]: i64, %[[VAL_110:.*]]: i64):
// CHECK:             %[[VAL_111:.*]] = arith.muli %[[VAL_108]], %[[VAL_109]] : i64
// CHECK:             linalg.yield %[[VAL_111]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_112:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_113:.*]] = arith.index_cast %[[VAL_112]] : index to i32
// CHECK:           %[[VAL_114:.*]] = linalg.fill ins(%[[VAL_113]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_115:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_63]], %[[VAL_114]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_63]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_116:.*]]: i32, %[[VAL_117:.*]]: i32, %[[VAL_118:.*]]: i32):
// CHECK:             %[[VAL_119:.*]] = arith.muli %[[VAL_116]], %[[VAL_117]] : i32
// CHECK:             linalg.yield %[[VAL_119]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_120:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_107]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_121:.*]]: i64, %[[VAL_122:.*]]: i32):
// CHECK:             %[[VAL_123:.*]] = arith.trunci %[[VAL_121]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_123]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_124:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_115]], %[[VAL_120]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_115]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_125:.*]]: i32, %[[VAL_126:.*]]: i32, %[[VAL_127:.*]]: i32):
// CHECK:             %[[VAL_128:.*]] = arith.addi %[[VAL_125]], %[[VAL_126]] : i32
// CHECK:             linalg.yield %[[VAL_128]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_129:.*]] = arith.index_cast %[[VAL_105]] : index to i32
// CHECK:           %[[VAL_130:.*]] = linalg.fill ins(%[[VAL_129]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_131:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_130]], %[[VAL_124]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_130]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_132:.*]]: i32, %[[VAL_133:.*]]: i32, %[[VAL_134:.*]]: i32):
// CHECK:             %[[VAL_135:.*]] = arith.addi %[[VAL_132]], %[[VAL_133]] : i32
// CHECK:             linalg.yield %[[VAL_135]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_136:.*]] = tensor.collapse_shape %[[VAL_131]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_137:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_138:.*]] = tensor.extract %[[VAL_136]]{{\[}}%[[VAL_137]]] : tensor<16xi32>
// CHECK:             %[[VAL_139:.*]] = arith.index_cast %[[VAL_138]] : i32 to index
// CHECK:             %[[VAL_140:.*]] = tensor.extract_slice %[[VAL_100]]{{\[}}%[[VAL_137]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_141:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_139]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_142:.*]] = memref.subview %[[VAL_141]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_140]] in writable %[[VAL_142]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  tt.func public @row_gather2(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<1x16xi32>
    %cst_0 = arith.constant dense<8> : tensor<16x1xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id z : i32
    %1 = arith.muli %arg5, %c8_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %4 = arith.remsi %3, %cst_0 : tensor<16x1xi32>
    %5 = arith.muli %arg3, %c8_i32 : i32
    %6 = tt.splat %5 : i32 -> tensor<16x1xi32>
    %7 = arith.addi %4, %6 : tensor<16x1xi32>
    %8 = arith.divsi %3, %cst_0 : tensor<16x1xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<16x1xi32>
    %10 = arith.addi %8, %9 : tensor<16x1xi32>
    %11 = arith.extsi %arg2 : i32 to i64
    %12 = arith.muli %11, %arg7 : i64
    %13 = tt.addptr %arg0, %12 : !tt.ptr<f32>, i64
    %14 = arith.extsi %10 : tensor<16x1xi32> to tensor<16x1xi64>
    %15 = tt.splat %arg6 : i64 -> tensor<16x1xi64>
    %16 = arith.muli %14, %15 : tensor<16x1xi64>
    %17 = tt.splat %13 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %18 = arith.extsi %7 : tensor<16x1xi32> to tensor<16x1xi64>
    %19 = tt.splat %arg8 : i64 -> tensor<16x1xi64>
    %20 = arith.muli %18, %19 : tensor<16x1xi64>
    %21 = arith.addi %16, %20 : tensor<16x1xi64>
    %22 = tt.addptr %17, %21 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %23 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %24 = tt.broadcast %22 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x16xi32> -> tensor<16x16xi32>
    %26 = tt.addptr %24, %25 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %27 = arith.cmpi slt, %23, %cst : tensor<1x16xi32>
    %28 = tt.splat %1 : i32 -> tensor<16x1xi32>
    %29 = arith.cmpi slt, %3, %28 : tensor<16x1xi32>
    %30 = tt.broadcast %27 : tensor<1x16xi1> -> tensor<16x16xi1>
    %31 = tt.broadcast %29 : tensor<16x1xi1> -> tensor<16x16xi1>
    %32 = arith.andi %30, %31 : tensor<16x16xi1>
    %33 = tt.load %26, %32 : tensor<16x16x!tt.ptr<f32>>
    %34 = arith.muli %11, %arg10 : i64
    %35 = arith.extsi %0 : i32 to i64
    %36 = arith.muli %35, %arg12 : i64
    %37 = arith.addi %34, %36 : i64
    %38 = tt.addptr %arg1, %37 : !tt.ptr<f32>, i64
    %39 = tt.splat %arg9 : i64 -> tensor<16x1xi64>
    %40 = arith.muli %14, %39 : tensor<16x1xi64>
    %41 = tt.splat %38 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %42 = tt.splat %arg11 : i64 -> tensor<16x1xi64>
    %43 = arith.muli %18, %42 : tensor<16x1xi64>
    %44 = arith.addi %40, %43 : tensor<16x1xi64>
    %45 = tt.addptr %41, %44 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %46 = tt.broadcast %45 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %47 = tt.addptr %46, %25 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %47, %33, %32 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}