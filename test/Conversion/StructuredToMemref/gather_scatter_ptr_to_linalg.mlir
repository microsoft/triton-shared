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
// CHECK:           %[[VAL_30:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_31:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_30]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_32:.*]]: i32):
// CHECK:             %[[VAL_33:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_34:.*]] = arith.index_cast %[[VAL_33]] : index to i32
// CHECK:             linalg.yield %[[VAL_34]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_35:.*]] = tensor.expand_shape %[[VAL_31]] {{\[\[}}0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
// CHECK:           %[[VAL_36:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_35]], %[[VAL_29]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_35]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_37:.*]]: i32, %[[VAL_38:.*]]: i32, %[[VAL_39:.*]]: i32):
// CHECK:             %[[VAL_40:.*]] = arith.remsi %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:             linalg.yield %[[VAL_40]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_41:.*]] = arith.muli %[[VAL_3]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_42:.*]] = linalg.fill ins(%[[VAL_41]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_43:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_36]], %[[VAL_42]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_36]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: i32):
// CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_44]], %[[VAL_45]] : i32
// CHECK:             linalg.yield %[[VAL_47]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_48:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_35]], %[[VAL_29]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_35]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_49:.*]]: i32, %[[VAL_50:.*]]: i32, %[[VAL_51:.*]]: i32):
// CHECK:             %[[VAL_52:.*]] = arith.divsi %[[VAL_49]], %[[VAL_50]] : i32
// CHECK:             linalg.yield %[[VAL_52]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_53:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_54:.*]] = arith.muli %[[VAL_53]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_55:.*]] = arith.index_cast %[[VAL_54]] : i64 to index
// CHECK:           %[[VAL_56:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_43]] : tensor<16x1xi32>) outs(%[[VAL_27]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_57:.*]]: i32, %[[VAL_58:.*]]: i64):
// CHECK:             %[[VAL_59:.*]] = arith.extsi %[[VAL_57]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_59]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_60:.*]] = linalg.fill ins(%[[VAL_8]] : i64) outs(%[[VAL_27]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_61:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_56]], %[[VAL_60]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_56]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_62:.*]]: i64, %[[VAL_63:.*]]: i64, %[[VAL_64:.*]]: i64):
// CHECK:             %[[VAL_65:.*]] = arith.muli %[[VAL_62]], %[[VAL_63]] : i64
// CHECK:             linalg.yield %[[VAL_65]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_66:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_67:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_48]], %[[VAL_66]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_48]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_68:.*]]: i32, %[[VAL_69:.*]]: i32, %[[VAL_70:.*]]: i32):
// CHECK:             %[[VAL_71:.*]] = arith.addi %[[VAL_68]], %[[VAL_69]] : i32
// CHECK:             linalg.yield %[[VAL_71]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_72:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_73:.*]] = arith.index_cast %[[VAL_72]] : index to i32
// CHECK:           %[[VAL_74:.*]] = linalg.fill ins(%[[VAL_73]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_75:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_67]], %[[VAL_74]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_67]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_76:.*]]: i32, %[[VAL_77:.*]]: i32, %[[VAL_78:.*]]: i32):
// CHECK:             %[[VAL_79:.*]] = arith.muli %[[VAL_76]], %[[VAL_77]] : i32
// CHECK:             linalg.yield %[[VAL_79]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_80:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_61]], %[[VAL_28]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_61]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_81:.*]]: i64, %[[VAL_82:.*]]: i64, %[[VAL_83:.*]]: i64):
// CHECK:             %[[VAL_84:.*]] = arith.remsi %[[VAL_81]], %[[VAL_82]] : i64
// CHECK:             linalg.yield %[[VAL_84]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_85:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_80]] : tensor<16x1xi64>) outs(%[[VAL_25]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_86:.*]]: i64, %[[VAL_87:.*]]: i32):
// CHECK:             %[[VAL_88:.*]] = arith.trunci %[[VAL_86]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_88]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_89:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_75]], %[[VAL_85]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_75]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_90:.*]]: i32, %[[VAL_91:.*]]: i32, %[[VAL_92:.*]]: i32):
// CHECK:             %[[VAL_93:.*]] = arith.addi %[[VAL_90]], %[[VAL_91]] : i32
// CHECK:             linalg.yield %[[VAL_93]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_94:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_89]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_89]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_95:.*]]: i32, %[[VAL_96:.*]]: i32, %[[VAL_97:.*]]: i32):
// CHECK:             %[[VAL_98:.*]] = arith.remsi %[[VAL_95]], %[[VAL_96]] : i32
// CHECK:             linalg.yield %[[VAL_98]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_99:.*]] = arith.index_cast %[[VAL_55]] : index to i32
// CHECK:           %[[VAL_100:.*]] = linalg.fill ins(%[[VAL_99]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_101:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_100]], %[[VAL_94]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_100]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_102:.*]]: i32, %[[VAL_103:.*]]: i32, %[[VAL_104:.*]]: i32):
// CHECK:             %[[VAL_105:.*]] = arith.addi %[[VAL_102]], %[[VAL_103]] : i32
// CHECK:             linalg.yield %[[VAL_105]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_106:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_101]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_101]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_107:.*]]: i32, %[[VAL_108:.*]]: i32, %[[VAL_109:.*]]: i32):
// CHECK:             %[[VAL_110:.*]] = arith.remsi %[[VAL_107]], %[[VAL_108]] : i32
// CHECK:             linalg.yield %[[VAL_110]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_111:.*]] = tensor.collapse_shape %[[VAL_106]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_112:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           scf.for %[[VAL_113:.*]] = %[[VAL_23]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_114:.*]] = tensor.extract %[[VAL_111]]{{\[}}%[[VAL_113]]] : tensor<16xi32>
// CHECK:             %[[VAL_115:.*]] = arith.index_cast %[[VAL_114]] : i32 to index
// CHECK:             %[[VAL_116:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_115]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_117:.*]] = memref.subview %[[VAL_116]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_118:.*]] = memref.subview %[[VAL_112]]{{\[}}%[[VAL_113]], 0] [1, 8] [1, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_117]], %[[VAL_118]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_119:.*]] = bufferization.to_tensor %[[VAL_112]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_120:.*]] = arith.muli %[[VAL_53]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_121:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_122:.*]] = arith.muli %[[VAL_121]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_123:.*]] = arith.addi %[[VAL_120]], %[[VAL_122]] : i64
// CHECK:           %[[VAL_124:.*]] = arith.index_cast %[[VAL_123]] : i64 to index
// CHECK:           %[[VAL_125:.*]] = linalg.fill ins(%[[VAL_11]] : i64) outs(%[[VAL_27]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_126:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_56]], %[[VAL_125]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_56]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_127:.*]]: i64, %[[VAL_128:.*]]: i64, %[[VAL_129:.*]]: i64):
// CHECK:             %[[VAL_130:.*]] = arith.muli %[[VAL_127]], %[[VAL_128]] : i64
// CHECK:             linalg.yield %[[VAL_130]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_131:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_132:.*]] = arith.index_cast %[[VAL_131]] : index to i32
// CHECK:           %[[VAL_133:.*]] = linalg.fill ins(%[[VAL_132]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_134:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_67]], %[[VAL_133]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_67]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_135:.*]]: i32, %[[VAL_136:.*]]: i32, %[[VAL_137:.*]]: i32):
// CHECK:             %[[VAL_138:.*]] = arith.muli %[[VAL_135]], %[[VAL_136]] : i32
// CHECK:             linalg.yield %[[VAL_138]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_139:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_126]], %[[VAL_28]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_126]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_140:.*]]: i64, %[[VAL_141:.*]]: i64, %[[VAL_142:.*]]: i64):
// CHECK:             %[[VAL_143:.*]] = arith.remsi %[[VAL_140]], %[[VAL_141]] : i64
// CHECK:             linalg.yield %[[VAL_143]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_144:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_139]] : tensor<16x1xi64>) outs(%[[VAL_25]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_145:.*]]: i64, %[[VAL_146:.*]]: i32):
// CHECK:             %[[VAL_147:.*]] = arith.trunci %[[VAL_145]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_147]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_148:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_134]], %[[VAL_144]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_134]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_149:.*]]: i32, %[[VAL_150:.*]]: i32, %[[VAL_151:.*]]: i32):
// CHECK:             %[[VAL_152:.*]] = arith.addi %[[VAL_149]], %[[VAL_150]] : i32
// CHECK:             linalg.yield %[[VAL_152]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_153:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_148]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_148]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_154:.*]]: i32, %[[VAL_155:.*]]: i32, %[[VAL_156:.*]]: i32):
// CHECK:             %[[VAL_157:.*]] = arith.remsi %[[VAL_154]], %[[VAL_155]] : i32
// CHECK:             linalg.yield %[[VAL_157]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_158:.*]] = arith.index_cast %[[VAL_124]] : index to i32
// CHECK:           %[[VAL_159:.*]] = linalg.fill ins(%[[VAL_158]] : i32) outs(%[[VAL_25]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_160:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_159]], %[[VAL_153]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_159]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_161:.*]]: i32, %[[VAL_162:.*]]: i32, %[[VAL_163:.*]]: i32):
// CHECK:             %[[VAL_164:.*]] = arith.addi %[[VAL_161]], %[[VAL_162]] : i32
// CHECK:             linalg.yield %[[VAL_164]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_165:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_160]], %[[VAL_26]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_160]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_166:.*]]: i32, %[[VAL_167:.*]]: i32, %[[VAL_168:.*]]: i32):
// CHECK:             %[[VAL_169:.*]] = arith.remsi %[[VAL_166]], %[[VAL_167]] : i32
// CHECK:             linalg.yield %[[VAL_169]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_170:.*]] = tensor.collapse_shape %[[VAL_165]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_171:.*]] = %[[VAL_23]] to %[[VAL_22]] step %[[VAL_19]] {
// CHECK:             %[[VAL_172:.*]] = tensor.extract %[[VAL_170]]{{\[}}%[[VAL_171]]] : tensor<16xi32>
// CHECK:             %[[VAL_173:.*]] = arith.index_cast %[[VAL_172]] : i32 to index
// CHECK:             %[[VAL_174:.*]] = tensor.extract_slice %[[VAL_119]]{{\[}}%[[VAL_171]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_175:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_173]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_176:.*]] = memref.subview %[[VAL_175]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_177:.*]] = memref.cast %[[VAL_176]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[?, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_174]] in writable %[[VAL_177]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[?, ?], offset: ?>>) -> ()
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