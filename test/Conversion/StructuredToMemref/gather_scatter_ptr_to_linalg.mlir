// RUN: triton-shared-opt --triton-to-linalg-experimental --cse %s | FileCheck %s

// RUN: triton-shared-opt --triton-to-linalg-experimental="enable-make-gather-scatter=false" %s | FileCheck %s --check-prefix=NO_GATHER

// Make sure no extract_slice when enable-make-gather-scatter=false..
// NO_GATHER-LABLE: func.func @row_gather2(
// NO_GATHER-NOT: extract_slice


// Make sure extract_slice is generated correctly.

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
// CHECK:           %[[VAL_76:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_71]], %[[VAL_70]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_71]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_77:.*]]: i32, %[[VAL_78:.*]]: i32, %[[VAL_79:.*]]: i32):
// CHECK:             %[[VAL_80:.*]] = arith.muli %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:             linalg.yield %[[VAL_80]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_81:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_57]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_82:.*]]: i64, %[[VAL_83:.*]]: i32):
// CHECK:             %[[VAL_84:.*]] = arith.trunci %[[VAL_82]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_84]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_85:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_76]], %[[VAL_81]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_76]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_86:.*]]: i32, %[[VAL_87:.*]]: i32, %[[VAL_88:.*]]: i32):
// CHECK:             %[[VAL_89:.*]] = arith.addi %[[VAL_86]], %[[VAL_87]] : i32
// CHECK:             linalg.yield %[[VAL_89]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_90:.*]] = arith.index_cast %[[VAL_50]] : index to i32
// CHECK:           %[[VAL_91:.*]] = linalg.fill ins(%[[VAL_90]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_92:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_91]], %[[VAL_85]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_91]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_93:.*]]: i32, %[[VAL_94:.*]]: i32, %[[VAL_95:.*]]: i32):
// CHECK:             %[[VAL_96:.*]] = arith.addi %[[VAL_93]], %[[VAL_94]] : i32
// CHECK:             linalg.yield %[[VAL_96]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_97:.*]] = tensor.collapse_shape %[[VAL_92]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_98:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           scf.for %[[VAL_99:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_100:.*]] = tensor.extract %[[VAL_97]]{{\[}}%[[VAL_99]]] : tensor<16xi32>
// CHECK:             %[[VAL_101:.*]] = arith.index_cast %[[VAL_100]] : i32 to index
// CHECK:             %[[VAL_102:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_101]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_103:.*]] = memref.subview %[[VAL_102]][0, 0] [1, 8] [16, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             %[[VAL_104:.*]] = memref.subview %[[VAL_98]]{{\[}}%[[VAL_99]], 0] [1, 8] [16, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[256, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_103]], %[[VAL_104]] : memref<1x8xf32, strided<[16, 1], offset: ?>> to memref<1x8xf32, strided<[256, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_105:.*]] = bufferization.to_tensor %[[VAL_98]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_106:.*]] = arith.muli %[[VAL_48]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_107:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_108:.*]] = arith.muli %[[VAL_107]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_109:.*]] = arith.addi %[[VAL_106]], %[[VAL_108]] : i64
// CHECK:           %[[VAL_110:.*]] = arith.index_cast %[[VAL_109]] : i64 to index
// CHECK:           %[[VAL_111:.*]] = linalg.fill ins(%[[VAL_11]] : i64) outs(%[[VAL_51]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_112:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_52]], %[[VAL_111]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_52]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_113:.*]]: i64, %[[VAL_114:.*]]: i64, %[[VAL_115:.*]]: i64):
// CHECK:             %[[VAL_116:.*]] = arith.muli %[[VAL_113]], %[[VAL_114]] : i64
// CHECK:             linalg.yield %[[VAL_116]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_117:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_118:.*]] = arith.index_cast %[[VAL_117]] : index to i32
// CHECK:           %[[VAL_119:.*]] = linalg.fill ins(%[[VAL_118]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_120:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_63]], %[[VAL_119]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_63]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_121:.*]]: i32, %[[VAL_122:.*]]: i32, %[[VAL_123:.*]]: i32):
// CHECK:             %[[VAL_124:.*]] = arith.muli %[[VAL_121]], %[[VAL_122]] : i32
// CHECK:             linalg.yield %[[VAL_124]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_125:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_120]], %[[VAL_119]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_120]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_126:.*]]: i32, %[[VAL_127:.*]]: i32, %[[VAL_128:.*]]: i32):
// CHECK:             %[[VAL_129:.*]] = arith.muli %[[VAL_126]], %[[VAL_127]] : i32
// CHECK:             linalg.yield %[[VAL_129]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_130:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_112]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_131:.*]]: i64, %[[VAL_132:.*]]: i32):
// CHECK:             %[[VAL_133:.*]] = arith.trunci %[[VAL_131]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_133]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_134:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_125]], %[[VAL_130]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_125]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_135:.*]]: i32, %[[VAL_136:.*]]: i32, %[[VAL_137:.*]]: i32):
// CHECK:             %[[VAL_138:.*]] = arith.addi %[[VAL_135]], %[[VAL_136]] : i32
// CHECK:             linalg.yield %[[VAL_138]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_139:.*]] = arith.index_cast %[[VAL_110]] : index to i32
// CHECK:           %[[VAL_140:.*]] = linalg.fill ins(%[[VAL_139]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_141:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_140]], %[[VAL_134]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_140]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_142:.*]]: i32, %[[VAL_143:.*]]: i32, %[[VAL_144:.*]]: i32):
// CHECK:             %[[VAL_145:.*]] = arith.addi %[[VAL_142]], %[[VAL_143]] : i32
// CHECK:             linalg.yield %[[VAL_145]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_146:.*]] = tensor.collapse_shape %[[VAL_141]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_147:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_148:.*]] = tensor.extract %[[VAL_146]]{{\[}}%[[VAL_147]]] : tensor<16xi32>
// CHECK:             %[[VAL_149:.*]] = arith.index_cast %[[VAL_148]] : i32 to index
// CHECK:             %[[VAL_150:.*]] = tensor.extract_slice %[[VAL_105]]{{\[}}%[[VAL_147]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_151:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_149]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_152:.*]] = memref.subview %[[VAL_151]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_150]] in writable %[[VAL_152]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[1, 1], offset: ?>>) -> ()
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