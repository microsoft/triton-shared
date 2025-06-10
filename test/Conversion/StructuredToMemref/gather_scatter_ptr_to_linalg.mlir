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
// CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_22:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_25:.*]] = tensor.empty() : tensor<16x1xi32>
// CHECK:           %[[VAL_26:.*]] = linalg.fill ins(%[[VAL_20]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_27:.*]] = tensor.empty() : tensor<16x1xi64>
// CHECK:           %[[VAL_28:.*]] = linalg.fill ins(%[[VAL_21]] : i64) outs(%[[VAL_27]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_29:.*]] = linalg.fill ins(%[[VAL_24]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_30:.*]] = arith.muli %[[VAL_5]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_31:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_32:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_31]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i32):
// CHECK:             %[[VAL_34:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_35:.*]] = arith.index_cast %[[VAL_34]] : index to i32
// CHECK:             linalg.yield %[[VAL_35]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_36:.*]] = tensor.expand_shape %[[VAL_32]] {{\[\[}}0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
// CHECK:           %[[VAL_37:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_36]], %[[VAL_29]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_36]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_38:.*]]: i32, %[[VAL_39:.*]]: i32, %[[VAL_40:.*]]: i32):
// CHECK:             %[[VAL_41:.*]] = arith.remsi %[[VAL_38]], %[[VAL_39]] : i32
// CHECK:             linalg.yield %[[VAL_41]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_42:.*]] = arith.muli %[[VAL_3]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_43:.*]] = linalg.fill ins(%[[VAL_42]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_44:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_37]], %[[VAL_43]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_37]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: i32, %[[VAL_47:.*]]: i32):
// CHECK:             %[[VAL_48:.*]] = arith.addi %[[VAL_45]], %[[VAL_46]] : i32
// CHECK:             linalg.yield %[[VAL_48]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_49:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_36]], %[[VAL_29]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_36]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_50:.*]]: i32, %[[VAL_51:.*]]: i32, %[[VAL_52:.*]]: i32):
// CHECK:             %[[VAL_53:.*]] = arith.divsi %[[VAL_50]], %[[VAL_51]] : i32
// CHECK:             linalg.yield %[[VAL_53]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_54:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_55:.*]] = arith.muli %[[VAL_54]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_56:.*]] = arith.index_cast %[[VAL_55]] : i64 to index
// CHECK:           %[[VAL_57:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_44]] : tensor<16x1xi32>) outs(%[[VAL_27]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_58:.*]]: i32, %[[VAL_59:.*]]: i64):
// CHECK:             %[[VAL_60:.*]] = arith.extsi %[[VAL_58]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_60]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_61:.*]] = linalg.fill ins(%[[VAL_8]] : i64) outs(%[[VAL_27]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_62:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_57]], %[[VAL_61]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_57]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_63:.*]]: i64, %[[VAL_64:.*]]: i64, %[[VAL_65:.*]]: i64):
// CHECK:             %[[VAL_66:.*]] = arith.muli %[[VAL_63]], %[[VAL_64]] : i64
// CHECK:             linalg.yield %[[VAL_66]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_67:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_68:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_49]], %[[VAL_67]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_49]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_69:.*]]: i32, %[[VAL_70:.*]]: i32, %[[VAL_71:.*]]: i32):
// CHECK:             %[[VAL_72:.*]] = arith.addi %[[VAL_69]], %[[VAL_70]] : i32
// CHECK:             linalg.yield %[[VAL_72]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_73:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_74:.*]] = arith.index_cast %[[VAL_73]] : index to i32
// CHECK:           %[[VAL_75:.*]] = linalg.fill ins(%[[VAL_74]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_76:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_68]], %[[VAL_75]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_68]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_77:.*]]: i32, %[[VAL_78:.*]]: i32, %[[VAL_79:.*]]: i32):
// CHECK:             %[[VAL_80:.*]] = arith.muli %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:             linalg.yield %[[VAL_80]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_81:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_62]], %[[VAL_28]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_62]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_82:.*]]: i64, %[[VAL_83:.*]]: i64, %[[VAL_84:.*]]: i64):
// CHECK:             %[[VAL_85:.*]] = arith.remsi %[[VAL_82]], %[[VAL_83]] : i64
// CHECK:             linalg.yield %[[VAL_85]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_86:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_81]] : tensor<16x1xi64>) outs(%[[VAL_25]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_87:.*]]: i64, %[[VAL_88:.*]]: i32):
// CHECK:             %[[VAL_89:.*]] = arith.trunci %[[VAL_87]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_89]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_90:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_76]], %[[VAL_86]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_76]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_91:.*]]: i32, %[[VAL_92:.*]]: i32, %[[VAL_93:.*]]: i32):
// CHECK:             %[[VAL_94:.*]] = arith.addi %[[VAL_91]], %[[VAL_92]] : i32
// CHECK:             linalg.yield %[[VAL_94]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_95:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_90]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_90]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_96:.*]]: i32, %[[VAL_97:.*]]: i32, %[[VAL_98:.*]]: i32):
// CHECK:             %[[VAL_99:.*]] = arith.remsi %[[VAL_96]], %[[VAL_97]] : i32
// CHECK:             linalg.yield %[[VAL_99]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_100:.*]] = arith.index_cast %[[VAL_56]] : index to i32
// CHECK:           %[[VAL_101:.*]] = linalg.fill ins(%[[VAL_100]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_102:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_101]], %[[VAL_95]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_101]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_103:.*]]: i32, %[[VAL_104:.*]]: i32, %[[VAL_105:.*]]: i32):
// CHECK:             %[[VAL_106:.*]] = arith.addi %[[VAL_103]], %[[VAL_104]] : i32
// CHECK:             linalg.yield %[[VAL_106]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_107:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_102]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_102]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_108:.*]]: i32, %[[VAL_109:.*]]: i32, %[[VAL_110:.*]]: i32):
// CHECK:             %[[VAL_111:.*]] = arith.remsi %[[VAL_108]], %[[VAL_109]] : i32
// CHECK:             linalg.yield %[[VAL_111]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_112:.*]] = tensor.collapse_shape %[[VAL_107]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_113:.*]] = arith.index_cast %[[VAL_30]] : i32 to index
// CHECK:           %[[VAL_114:.*]] = arith.minsi %[[VAL_113]], %[[VAL_22]] : index
// CHECK:           %[[VAL_115:.*]] = arith.maxsi %[[VAL_114]], %[[VAL_23]] : index
// CHECK:           %[[VAL_116:.*]] = arith.minsi %[[VAL_115]], %[[VAL_22]] : index
// CHECK:           %[[VAL_117:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           %[[VAL_118:.*]] = arith.minsi %[[VAL_116]], %[[VAL_22]] : index
// CHECK:           scf.for %[[VAL_119:.*]] = %[[VAL_23]] to %[[VAL_118]] step %[[VAL_19]] {
// CHECK:             %[[VAL_120:.*]] = tensor.extract %[[VAL_112]]{{\[}}%[[VAL_119]]] : tensor<16xi32>
// CHECK:             %[[VAL_121:.*]] = arith.index_cast %[[VAL_120]] : i32 to index
// CHECK:             %[[VAL_122:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_121]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_123:.*]] = memref.subview %[[VAL_122]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_124:.*]] = memref.subview %[[VAL_117]]{{\[}}%[[VAL_119]], 0] [1, 8] [1, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_123]], %[[VAL_124]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_125:.*]] = bufferization.to_tensor %[[VAL_117]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_126:.*]] = arith.muli %[[VAL_54]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_127:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_128:.*]] = arith.muli %[[VAL_127]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_129:.*]] = arith.addi %[[VAL_126]], %[[VAL_128]] : i64
// CHECK:           %[[VAL_130:.*]] = arith.index_cast %[[VAL_129]] : i64 to index
// CHECK:           %[[VAL_131:.*]] = linalg.fill ins(%[[VAL_11]] : i64) outs(%[[VAL_27]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_132:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_57]], %[[VAL_131]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_57]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_133:.*]]: i64, %[[VAL_134:.*]]: i64, %[[VAL_135:.*]]: i64):
// CHECK:             %[[VAL_136:.*]] = arith.muli %[[VAL_133]], %[[VAL_134]] : i64
// CHECK:             linalg.yield %[[VAL_136]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_137:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_138:.*]] = arith.index_cast %[[VAL_137]] : index to i32
// CHECK:           %[[VAL_139:.*]] = linalg.fill ins(%[[VAL_138]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_140:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_68]], %[[VAL_139]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_68]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_141:.*]]: i32, %[[VAL_142:.*]]: i32, %[[VAL_143:.*]]: i32):
// CHECK:             %[[VAL_144:.*]] = arith.muli %[[VAL_141]], %[[VAL_142]] : i32
// CHECK:             linalg.yield %[[VAL_144]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_145:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_132]], %[[VAL_28]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_132]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_146:.*]]: i64, %[[VAL_147:.*]]: i64, %[[VAL_148:.*]]: i64):
// CHECK:             %[[VAL_149:.*]] = arith.remsi %[[VAL_146]], %[[VAL_147]] : i64
// CHECK:             linalg.yield %[[VAL_149]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_150:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_145]] : tensor<16x1xi64>) outs(%[[VAL_25]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_151:.*]]: i64, %[[VAL_152:.*]]: i32):
// CHECK:             %[[VAL_153:.*]] = arith.trunci %[[VAL_151]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_153]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_154:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_140]], %[[VAL_150]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_140]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_155:.*]]: i32, %[[VAL_156:.*]]: i32, %[[VAL_157:.*]]: i32):
// CHECK:             %[[VAL_158:.*]] = arith.addi %[[VAL_155]], %[[VAL_156]] : i32
// CHECK:             linalg.yield %[[VAL_158]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_159:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_154]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_154]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_160:.*]]: i32, %[[VAL_161:.*]]: i32, %[[VAL_162:.*]]: i32):
// CHECK:             %[[VAL_163:.*]] = arith.remsi %[[VAL_160]], %[[VAL_161]] : i32
// CHECK:             linalg.yield %[[VAL_163]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_164:.*]] = arith.index_cast %[[VAL_130]] : index to i32
// CHECK:           %[[VAL_165:.*]] = linalg.fill ins(%[[VAL_164]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_166:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_165]], %[[VAL_159]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_165]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_167:.*]]: i32, %[[VAL_168:.*]]: i32, %[[VAL_169:.*]]: i32):
// CHECK:             %[[VAL_170:.*]] = arith.addi %[[VAL_167]], %[[VAL_168]] : i32
// CHECK:             linalg.yield %[[VAL_170]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_171:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_166]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_166]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_172:.*]]: i32, %[[VAL_173:.*]]: i32, %[[VAL_174:.*]]: i32):
// CHECK:             %[[VAL_175:.*]] = arith.remsi %[[VAL_172]], %[[VAL_173]] : i32
// CHECK:             linalg.yield %[[VAL_175]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_176:.*]] = tensor.collapse_shape %[[VAL_171]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_177:.*]] = %[[VAL_23]] to %[[VAL_118]] step %[[VAL_19]] {
// CHECK:             %[[VAL_178:.*]] = tensor.extract %[[VAL_176]]{{\[}}%[[VAL_177]]] : tensor<16xi32>
// CHECK:             %[[VAL_179:.*]] = arith.index_cast %[[VAL_178]] : i32 to index
// CHECK:             %[[VAL_180:.*]] = tensor.extract_slice %[[VAL_125]]{{\[}}%[[VAL_177]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_181:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_179]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_182:.*]] = memref.subview %[[VAL_181]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_183:.*]] = memref.cast %[[VAL_182]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[?, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_180]] in writable %[[VAL_183]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return

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