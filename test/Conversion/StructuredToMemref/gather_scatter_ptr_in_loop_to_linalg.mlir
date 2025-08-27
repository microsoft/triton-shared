// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure extract_slice and subview is generated with correct offsets and strides.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @row_gather3(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_19:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_20:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_21:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_25:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_26:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_27:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_28:.*]] = arith.constant 64 : i32
// CHECK:           %[[VAL_29:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[VAL_30:.*]] = linalg.fill ins(%[[VAL_26]] : f32) outs(%[[VAL_29]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:           %[[VAL_31:.*]] = arith.muli %[[VAL_19]], %[[VAL_28]] : i32
// CHECK:           %[[VAL_32:.*]] = tensor.empty() : tensor<64xi32>
// CHECK:           %[[VAL_33:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_32]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_34:.*]]: i32):
// CHECK:             %[[VAL_35:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : index to i32
// CHECK:             linalg.yield %[[VAL_36]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_37:.*]] = linalg.fill ins(%[[VAL_31]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_38:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_37]], %[[VAL_33]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_37]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_39:.*]]: i32, %[[VAL_40:.*]]: i32, %[[VAL_41:.*]]: i32):
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_39]], %[[VAL_40]] : i32
// CHECK:             linalg.yield %[[VAL_42]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_43:.*]] = arith.muli %[[VAL_20]], %[[VAL_28]] : i32
// CHECK:           %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : i32 to index
// CHECK:           %[[VAL_45:.*]] = arith.muli %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_46:.*]] = linalg.fill ins(%[[VAL_45]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_47:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_38]], %[[VAL_46]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_38]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_48:.*]]: i32, %[[VAL_49:.*]]: i32, %[[VAL_50:.*]]: i32):
// CHECK:             %[[VAL_51:.*]] = arith.divsi %[[VAL_48]], %[[VAL_49]] : i32
// CHECK:             linalg.yield %[[VAL_51]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_52:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_38]], %[[VAL_46]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_38]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_53:.*]]: i32, %[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i32):
// CHECK:             %[[VAL_56:.*]] = arith.remsi %[[VAL_53]], %[[VAL_54]] : i32
// CHECK:             linalg.yield %[[VAL_56]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_57:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_58:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_52]], %[[VAL_57]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_52]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_59:.*]]: i32, %[[VAL_60:.*]]: i32, %[[VAL_61:.*]]: i32):
// CHECK:             %[[VAL_62:.*]] = arith.divsi %[[VAL_59]], %[[VAL_60]] : i32
// CHECK:             linalg.yield %[[VAL_62]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_63:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_52]], %[[VAL_57]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_52]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_64:.*]]: i32, %[[VAL_65:.*]]: i32, %[[VAL_66:.*]]: i32):
// CHECK:             %[[VAL_67:.*]] = arith.remsi %[[VAL_64]], %[[VAL_65]] : i32
// CHECK:             linalg.yield %[[VAL_67]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_68:.*]] = scf.for %[[VAL_69:.*]] = %[[VAL_27]] to %[[VAL_6]] step %[[VAL_24]] iter_args(%[[VAL_70:.*]] = %[[VAL_30]]) -> (tensor<64x64xf32>)  : i32 {
// CHECK:             %[[VAL_71:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_72:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_47]], %[[VAL_71]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_47]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_73:.*]]: i32, %[[VAL_74:.*]]: i32, %[[VAL_75:.*]]: i32):
// CHECK:               %[[VAL_76:.*]] = arith.muli %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:               linalg.yield %[[VAL_76]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_77:.*]] = linalg.fill ins(%[[VAL_8]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_78:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_58]], %[[VAL_77]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_58]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_79:.*]]: i32, %[[VAL_80:.*]]: i32, %[[VAL_81:.*]]: i32):
// CHECK:               %[[VAL_82:.*]] = arith.muli %[[VAL_79]], %[[VAL_80]] : i32
// CHECK:               linalg.yield %[[VAL_82]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_83:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_72]], %[[VAL_78]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_72]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_84:.*]]: i32, %[[VAL_85:.*]]: i32, %[[VAL_86:.*]]: i32):
// CHECK:               %[[VAL_87:.*]] = arith.addi %[[VAL_84]], %[[VAL_85]] : i32
// CHECK:               linalg.yield %[[VAL_87]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_88:.*]] = linalg.fill ins(%[[VAL_9]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_89:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_63]], %[[VAL_88]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_63]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_90:.*]]: i32, %[[VAL_91:.*]]: i32, %[[VAL_92:.*]]: i32):
// CHECK:               %[[VAL_93:.*]] = arith.muli %[[VAL_90]], %[[VAL_91]] : i32
// CHECK:               linalg.yield %[[VAL_93]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_94:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_83]], %[[VAL_89]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_83]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_95:.*]]: i32, %[[VAL_96:.*]]: i32, %[[VAL_97:.*]]: i32):
// CHECK:               %[[VAL_98:.*]] = arith.addi %[[VAL_95]], %[[VAL_96]] : i32
// CHECK:               linalg.yield %[[VAL_98]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_99:.*]] = linalg.fill ins(%[[VAL_69]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_100:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_94]], %[[VAL_99]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_94]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_101:.*]]: i32, %[[VAL_102:.*]]: i32, %[[VAL_103:.*]]: i32):
// CHECK:               %[[VAL_104:.*]] = arith.addi %[[VAL_101]], %[[VAL_102]] : i32
// CHECK:               linalg.yield %[[VAL_104]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_105:.*]] = arith.index_cast %[[VAL_31]] : i32 to index
// CHECK:             %[[VAL_106:.*]] = arith.addi %[[VAL_105]], %[[VAL_25]] : index
// CHECK:             %[[VAL_107:.*]] = arith.index_cast %[[VAL_45]] : i32 to index
// CHECK:             %[[VAL_108:.*]] = arith.minsi %[[VAL_106]], %[[VAL_107]] : index
// CHECK:             %[[VAL_109:.*]] = arith.maxsi %[[VAL_108]], %[[VAL_105]] : index
// CHECK:             %[[VAL_110:.*]] = arith.subi %[[VAL_109]], %[[VAL_105]] : index
// CHECK:             %[[VAL_111:.*]] = memref.alloc() : memref<64xf32>
// CHECK:             %[[VAL_112:.*]] = arith.cmpi slt, %[[VAL_110]], %[[VAL_25]] : index
// CHECK:             scf.if %[[VAL_112]] {
// CHECK:               linalg.fill ins(%[[VAL_26]] : f32) outs(%[[VAL_111]] : memref<64xf32>)
// CHECK:             }
// CHECK:             %[[VAL_113:.*]] = arith.minsi %[[VAL_110]], %[[VAL_25]] : index
// CHECK:             scf.for %[[VAL_114:.*]] = %[[VAL_23]] to %[[VAL_113]] step %[[VAL_22]] {
// CHECK:               %[[VAL_115:.*]] = tensor.extract %[[VAL_100]]{{\[}}%[[VAL_114]]] : tensor<64xi32>
// CHECK:               %[[VAL_116:.*]] = arith.index_cast %[[VAL_115]] : i32 to index
// CHECK:               %[[VAL_117:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_116]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[VAL_118:.*]] = memref.subview %[[VAL_111]]{{\[}}%[[VAL_114]]] [1] [1] : memref<64xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[VAL_117]], %[[VAL_118]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             %[[VAL_119:.*]] = bufferization.to_tensor %[[VAL_111]] restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK:             %[[VAL_120:.*]] = arith.index_cast %[[VAL_12]] : i32 to index
// CHECK:             %[[VAL_121:.*]] = arith.muli %[[VAL_44]], %[[VAL_120]] : index
// CHECK:             %[[VAL_122:.*]] = linalg.fill ins(%[[VAL_10]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_123:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_58]], %[[VAL_122]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_58]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_124:.*]]: i32, %[[VAL_125:.*]]: i32, %[[VAL_126:.*]]: i32):
// CHECK:               %[[VAL_127:.*]] = arith.muli %[[VAL_124]], %[[VAL_125]] : i32
// CHECK:               linalg.yield %[[VAL_127]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_128:.*]] = linalg.fill ins(%[[VAL_11]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:             %[[VAL_129:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_63]], %[[VAL_128]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_63]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_130:.*]]: i32, %[[VAL_131:.*]]: i32, %[[VAL_132:.*]]: i32):
// CHECK:               %[[VAL_133:.*]] = arith.muli %[[VAL_130]], %[[VAL_131]] : i32
// CHECK:               linalg.yield %[[VAL_133]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_134:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_123]], %[[VAL_129]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_123]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_135:.*]]: i32, %[[VAL_136:.*]]: i32, %[[VAL_137:.*]]: i32):
// CHECK:               %[[VAL_138:.*]] = arith.addi %[[VAL_135]], %[[VAL_136]] : i32
// CHECK:               linalg.yield %[[VAL_138]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_139:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_134]], %[[VAL_99]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_134]] : tensor<64xi32>) {
// CHECK:             ^bb0(%[[VAL_140:.*]]: i32, %[[VAL_141:.*]]: i32, %[[VAL_142:.*]]: i32):
// CHECK:               %[[VAL_143:.*]] = arith.addi %[[VAL_140]], %[[VAL_141]] : i32
// CHECK:               linalg.yield %[[VAL_143]] : i32
// CHECK:             } -> tensor<64xi32>
// CHECK:             %[[VAL_144:.*]] = arith.addi %[[VAL_44]], %[[VAL_25]] : index
// CHECK:             %[[VAL_145:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:             %[[VAL_146:.*]] = arith.minsi %[[VAL_144]], %[[VAL_145]] : index
// CHECK:             %[[VAL_147:.*]] = arith.maxsi %[[VAL_146]], %[[VAL_44]] : index
// CHECK:             %[[VAL_148:.*]] = arith.subi %[[VAL_147]], %[[VAL_44]] : index
// CHECK:             %[[VAL_149:.*]] = arith.minsi %[[VAL_148]], %[[VAL_25]] : index
// CHECK:             %[[VAL_150:.*]] = memref.alloc() : memref<64x64xf32>
// CHECK:             %[[VAL_151:.*]] = arith.cmpi slt, %[[VAL_113]], %[[VAL_25]] : index
// CHECK:             %[[VAL_152:.*]] = arith.cmpi slt, %[[VAL_149]], %[[VAL_25]] : index
// CHECK:             %[[VAL_153:.*]] = arith.ori %[[VAL_151]], %[[VAL_152]] : i1
// CHECK:             scf.if %[[VAL_153]] {
// CHECK:               linalg.fill ins(%[[VAL_26]] : f32) outs(%[[VAL_150]] : memref<64x64xf32>)
// CHECK:             }
// CHECK:             %[[VAL_154:.*]] = arith.minsi %[[VAL_113]], %[[VAL_25]] : index
// CHECK:             scf.for %[[VAL_155:.*]] = %[[VAL_23]] to %[[VAL_154]] step %[[VAL_22]] {
// CHECK:               %[[VAL_156:.*]] = tensor.extract %[[VAL_139]]{{\[}}%[[VAL_155]]] : tensor<64xi32>
// CHECK:               %[[VAL_157:.*]] = arith.index_cast %[[VAL_156]] : i32 to index
// CHECK:               %[[VAL_158:.*]] = arith.addi %[[VAL_157]], %[[VAL_121]] : index
// CHECK:               %[[VAL_159:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_158]]], sizes: [1, 64], strides: [1, %[[VAL_120]]] : memref<*xf32> to memref<1x64xf32, strided<[1, ?], offset: ?>>
// CHECK:               %[[VAL_160:.*]] = memref.subview %[[VAL_159]][0, 0] [1, %[[VAL_149]]] [1, 1] : memref<1x64xf32, strided<[1, ?], offset: ?>> to memref<1x?xf32, strided<[1, ?], offset: ?>>
// CHECK:               %[[VAL_161:.*]] = memref.subview %[[VAL_150]]{{\[}}%[[VAL_155]], 0] [1, %[[VAL_149]]] [1, 1] : memref<64x64xf32> to memref<1x?xf32, strided<[64, 1], offset: ?>>
// CHECK:               memref.copy %[[VAL_160]], %[[VAL_161]] : memref<1x?xf32, strided<[1, ?], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
// CHECK:             }
// CHECK:             %[[VAL_162:.*]] = bufferization.to_tensor %[[VAL_150]] restrict writable : memref<64x64xf32> to tensor<64x64xf32>
// CHECK:             %[[VAL_163:.*]] = tensor.expand_shape %[[VAL_119]] {{\[\[}}0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
// CHECK:             %[[VAL_164:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_163]] : tensor<64x1xf32>) outs(%[[VAL_29]] : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:             ^bb0(%[[VAL_165:.*]]: f32, %[[VAL_166:.*]]: f32):
// CHECK:               linalg.yield %[[VAL_165]] : f32
// CHECK:             } -> tensor<64x64xf32>
// CHECK:             %[[VAL_167:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_164]], %[[VAL_162]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[VAL_164]] : tensor<64x64xf32>) {
// CHECK:             ^bb0(%[[VAL_168:.*]]: f32, %[[VAL_169:.*]]: f32, %[[VAL_170:.*]]: f32):
// CHECK:               %[[VAL_171:.*]] = arith.mulf %[[VAL_168]], %[[VAL_169]] : f32
// CHECK:               linalg.yield %[[VAL_171]] : f32
// CHECK:             } -> tensor<64x64xf32>
// CHECK:             %[[VAL_172:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_70]], %[[VAL_167]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[VAL_70]] : tensor<64x64xf32>) {
// CHECK:             ^bb0(%[[VAL_173:.*]]: f32, %[[VAL_174:.*]]: f32, %[[VAL_175:.*]]: f32):
// CHECK:               %[[VAL_176:.*]] = arith.addf %[[VAL_173]], %[[VAL_174]] : f32
// CHECK:               linalg.yield %[[VAL_176]] : f32
// CHECK:             } -> tensor<64x64xf32>
// CHECK:             scf.yield %[[VAL_172]] : tensor<64x64xf32>
// CHECK:           }
// CHECK:           %[[VAL_177:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_178:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_47]], %[[VAL_177]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_47]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_179:.*]]: i32, %[[VAL_180:.*]]: i32, %[[VAL_181:.*]]: i32):
// CHECK:             %[[VAL_182:.*]] = arith.muli %[[VAL_179]], %[[VAL_180]] : i32
// CHECK:             linalg.yield %[[VAL_182]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_183:.*]] = linalg.fill ins(%[[VAL_14]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_184:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_58]], %[[VAL_183]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_58]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_185:.*]]: i32, %[[VAL_186:.*]]: i32, %[[VAL_187:.*]]: i32):
// CHECK:             %[[VAL_188:.*]] = arith.muli %[[VAL_185]], %[[VAL_186]] : i32
// CHECK:             linalg.yield %[[VAL_188]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_189:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_178]], %[[VAL_184]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_178]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_190:.*]]: i32, %[[VAL_191:.*]]: i32, %[[VAL_192:.*]]: i32):
// CHECK:             %[[VAL_193:.*]] = arith.addi %[[VAL_190]], %[[VAL_191]] : i32
// CHECK:             linalg.yield %[[VAL_193]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_194:.*]] = linalg.fill ins(%[[VAL_15]] : i32) outs(%[[VAL_32]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[VAL_195:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_63]], %[[VAL_194]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_63]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_196:.*]]: i32, %[[VAL_197:.*]]: i32, %[[VAL_198:.*]]: i32):
// CHECK:             %[[VAL_199:.*]] = arith.muli %[[VAL_196]], %[[VAL_197]] : i32
// CHECK:             linalg.yield %[[VAL_199]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_200:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_189]], %[[VAL_195]] : tensor<64xi32>, tensor<64xi32>) outs(%[[VAL_189]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_201:.*]]: i32, %[[VAL_202:.*]]: i32, %[[VAL_203:.*]]: i32):
// CHECK:             %[[VAL_204:.*]] = arith.addi %[[VAL_201]], %[[VAL_202]] : i32
// CHECK:             linalg.yield %[[VAL_204]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[VAL_205:.*]] = arith.index_cast %[[VAL_31]] : i32 to index
// CHECK:           %[[VAL_206:.*]] = arith.addi %[[VAL_205]], %[[VAL_25]] : index
// CHECK:           %[[VAL_207:.*]] = arith.index_cast %[[VAL_45]] : i32 to index
// CHECK:           %[[VAL_208:.*]] = arith.minsi %[[VAL_206]], %[[VAL_207]] : index
// CHECK:           %[[VAL_209:.*]] = arith.maxsi %[[VAL_208]], %[[VAL_205]] : index
// CHECK:           %[[VAL_210:.*]] = arith.subi %[[VAL_209]], %[[VAL_205]] : index
// CHECK:           %[[VAL_211:.*]] = arith.addi %[[VAL_44]], %[[VAL_25]] : index
// CHECK:           %[[VAL_212:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_213:.*]] = arith.minsi %[[VAL_211]], %[[VAL_212]] : index
// CHECK:           %[[VAL_214:.*]] = arith.maxsi %[[VAL_213]], %[[VAL_44]] : index
// CHECK:           %[[VAL_215:.*]] = arith.subi %[[VAL_214]], %[[VAL_44]] : index
// CHECK:           %[[VAL_216:.*]] = arith.minsi %[[VAL_210]], %[[VAL_25]] : index
// CHECK:           %[[VAL_217:.*]] = arith.minsi %[[VAL_215]], %[[VAL_25]] : index
// CHECK:           %[[VAL_218:.*]] = arith.minsi %[[VAL_216]], %[[VAL_25]] : index
// CHECK:           scf.for %[[VAL_219:.*]] = %[[VAL_23]] to %[[VAL_218]] step %[[VAL_22]] {
// CHECK:             %[[VAL_220:.*]] = tensor.extract %[[VAL_200]]{{\[}}%[[VAL_219]]] : tensor<64xi32>
// CHECK:             %[[VAL_221:.*]] = arith.index_cast %[[VAL_220]] : i32 to index
// CHECK:             %[[VAL_222:.*]] = tensor.extract_slice %[[VAL_68]]{{\[}}%[[VAL_219]], 0] [1, %[[VAL_217]]] [1, 1] : tensor<64x64xf32> to tensor<1x?xf32>
// CHECK:             %[[VAL_223:.*]] = arith.addi %[[VAL_221]], %[[VAL_44]] : index
// CHECK:             %[[VAL_224:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_223]]], sizes: [1, 64], strides: [1, 1] : memref<*xf32> to memref<1x64xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_225:.*]] = memref.subview %[[VAL_224]][0, 0] [1, %[[VAL_217]]] [1, 1] : memref<1x64xf32, strided<[1, 1], offset: ?>> to memref<1x?xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_226:.*]] = memref.cast %[[VAL_225]] : memref<1x?xf32, strided<[1, 1], offset: ?>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_222]] in writable %[[VAL_226]] : (tensor<1x?xf32>, memref<1x?xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return

module attributes {} {
  tt.func public @row_gather3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32 , %arg6: i32 , %arg7: i32 , %arg8: i32 , %arg9: i32 , %arg10: i32 , %arg11: i32 , %arg12: i32 , %arg13: i32 , %arg14: i32 , %arg15: i32 ) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.splat %2 : i32 -> tensor<64xi32>
    %5 = arith.addi %4, %3 : tensor<64xi32>
    %6 = arith.muli %1, %c64_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<64xi32>
    %8 = arith.addi %7, %3 : tensor<64xi32>
    %9 = arith.muli %arg3, %arg4 : i32
    %10 = tt.splat %9 : i32 -> tensor<64xi32>
    %11 = arith.cmpi slt, %5, %10 : tensor<64xi32>
    %12 = tt.splat %arg5 : i32 -> tensor<64xi32>
    %13 = arith.cmpi slt, %8, %12 : tensor<64xi32>
    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %15 = tt.broadcast %14 : tensor<64x1xi1> -> tensor<64x64xi1>
    %16 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %17 = tt.broadcast %16 : tensor<1x64xi1> -> tensor<64x64xi1>
    %18 = arith.andi %15, %17 : tensor<64x64xi1>
    %19 = arith.divsi %5, %10 : tensor<64xi32>
    %20 = arith.remsi %5, %10 : tensor<64xi32>
    %21 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %22 = arith.divsi %20, %21 : tensor<64xi32>
    %23 = arith.remsi %20, %21 : tensor<64xi32>
    %24 = scf.for %arg16 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg17 = %cst) -> (tensor<64x64xf32>)  : i32 {
      %40 = tt.splat %arg7 : i32 -> tensor<64xi32>
      %41 = arith.muli %19, %40 : tensor<64xi32>
      %42 = tt.splat %arg8 : i32 -> tensor<64xi32>
      %43 = arith.muli %22, %42 : tensor<64xi32>
      %44 = arith.addi %41, %43 : tensor<64xi32>
      %45 = tt.splat %arg9 : i32 -> tensor<64xi32>
      %46 = arith.muli %23, %45 : tensor<64xi32>
      %47 = arith.addi %44, %46 : tensor<64xi32>
      %48 = tt.splat %arg16 : i32 -> tensor<64xi32>
      %49 = arith.addi %47, %48 : tensor<64xi32>
      %50 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %51 = tt.addptr %50, %49 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %52 = tt.load %51, %11, %cst_0 : tensor<64x!tt.ptr<f32>>
      %53 = tt.splat %arg10 : i32 -> tensor<64xi32>
      %54 = arith.muli %22, %53 : tensor<64xi32>
      %55 = tt.splat %arg11 : i32 -> tensor<64xi32>
      %56 = arith.muli %23, %55 : tensor<64xi32>
      %57 = arith.addi %54, %56 : tensor<64xi32>
      %58 = arith.addi %57, %48 : tensor<64xi32>
      %59 = tt.expand_dims %58 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %60 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %61 = tt.splat %arg12 : i32 -> tensor<1x64xi32>
      %62 = arith.muli %60, %61 : tensor<1x64xi32>
      %63 = tt.broadcast %59 : tensor<64x1xi32> -> tensor<64x64xi32>
      %64 = tt.broadcast %62 : tensor<1x64xi32> -> tensor<64x64xi32>
      %65 = arith.addi %63, %64 : tensor<64x64xi32>
      %66 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
      %67 = tt.addptr %66, %65 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
      %68 = tt.load %67, %18, %cst : tensor<64x64x!tt.ptr<f32>>
      %69 = tt.expand_dims %52 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %70 = tt.broadcast %69 : tensor<64x1xf32> -> tensor<64x64xf32>
      %71 = arith.mulf %70, %68 : tensor<64x64xf32>
      %72 = arith.addf %arg17, %71 : tensor<64x64xf32>
      scf.yield %72 : tensor<64x64xf32>
    }
    %25 = tt.splat %arg13 : i32 -> tensor<64xi32>
    %26 = arith.muli %19, %25 : tensor<64xi32>
    %27 = tt.splat %arg14 : i32 -> tensor<64xi32>
    %28 = arith.muli %22, %27 : tensor<64xi32>
    %29 = arith.addi %26, %28 : tensor<64xi32>
    %30 = tt.splat %arg15 : i32 -> tensor<64xi32>
    %31 = arith.muli %23, %30 : tensor<64xi32>
    %32 = arith.addi %29, %31 : tensor<64xi32>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %34 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %35 = tt.broadcast %33 : tensor<64x1xi32> -> tensor<64x64xi32>
    %36 = tt.broadcast %34 : tensor<1x64xi32> -> tensor<64x64xi32>
    %37 = arith.addi %35, %36 : tensor<64x64xi32>
    %38 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %39 = tt.addptr %38, %37 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    tt.store %39, %24, %18 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}
