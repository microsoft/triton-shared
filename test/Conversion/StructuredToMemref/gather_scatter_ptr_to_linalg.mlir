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
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_5]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_26:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_26]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i32
// CHECK:             linalg.yield %[[VAL_30]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_31:.*]] = tensor.expand_shape %[[VAL_27]] {{\[\[}}0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
// CHECK:           %[[VAL_32:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_31]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i32, %[[VAL_35:.*]]: i32):
// CHECK:             %[[VAL_36:.*]] = arith.remsi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:             linalg.yield %[[VAL_36]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_37:.*]] = arith.muli %[[VAL_3]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_38:.*]] = linalg.fill ins(%[[VAL_37]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_39:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_32]], %[[VAL_38]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_32]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_40:.*]]: i32, %[[VAL_41:.*]]: i32, %[[VAL_42:.*]]: i32):
// CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:             linalg.yield %[[VAL_43]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_44:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_31]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: i32, %[[VAL_47:.*]]: i32):
// CHECK:             %[[VAL_48:.*]] = arith.divsi %[[VAL_45]], %[[VAL_46]] : i32
// CHECK:             linalg.yield %[[VAL_48]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_49:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_50:.*]] = arith.muli %[[VAL_49]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_51:.*]] = arith.index_cast %[[VAL_50]] : i64 to index
// CHECK:           %[[VAL_52:.*]] = tensor.empty() : tensor<16x1xi64>
// CHECK:           %[[VAL_53:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_39]] : tensor<16x1xi32>) outs(%[[VAL_52]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i64):
// CHECK:             %[[VAL_56:.*]] = arith.extsi %[[VAL_54]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_56]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_57:.*]] = linalg.fill ins(%[[VAL_8]] : i64) outs(%[[VAL_52]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_58:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_53]], %[[VAL_57]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_53]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_59:.*]]: i64, %[[VAL_60:.*]]: i64, %[[VAL_61:.*]]: i64):
// CHECK:             %[[VAL_62:.*]] = arith.muli %[[VAL_59]], %[[VAL_60]] : i64
// CHECK:             linalg.yield %[[VAL_62]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_63:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_64:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_44]], %[[VAL_63]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_44]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_65:.*]]: i32, %[[VAL_66:.*]]: i32, %[[VAL_67:.*]]: i32):
// CHECK:             %[[VAL_68:.*]] = arith.addi %[[VAL_65]], %[[VAL_66]] : i32
// CHECK:             linalg.yield %[[VAL_68]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_69:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_70:.*]] = arith.index_cast %[[VAL_69]] : index to i32
// CHECK:           %[[VAL_71:.*]] = linalg.fill ins(%[[VAL_70]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_72:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_64]], %[[VAL_71]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_64]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_73:.*]]: i32, %[[VAL_74:.*]]: i32, %[[VAL_75:.*]]: i32):
// CHECK:             %[[VAL_76:.*]] = arith.muli %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:             linalg.yield %[[VAL_76]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_77:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_72]], %[[VAL_71]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_72]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_78:.*]]: i32, %[[VAL_79:.*]]: i32, %[[VAL_80:.*]]: i32):
// CHECK:             %[[VAL_81:.*]] = arith.muli %[[VAL_78]], %[[VAL_79]] : i32
// CHECK:             linalg.yield %[[VAL_81]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_82:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_58]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_83:.*]]: i64, %[[VAL_84:.*]]: i32):
// CHECK:             %[[VAL_85:.*]] = arith.trunci %[[VAL_83]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_85]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_86:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_77]], %[[VAL_82]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_77]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_87:.*]]: i32, %[[VAL_88:.*]]: i32, %[[VAL_89:.*]]: i32):
// CHECK:             %[[VAL_90:.*]] = arith.addi %[[VAL_87]], %[[VAL_88]] : i32
// CHECK:             linalg.yield %[[VAL_90]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_91:.*]] = arith.index_cast %[[VAL_51]] : index to i32
// CHECK:           %[[VAL_92:.*]] = linalg.fill ins(%[[VAL_91]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_93:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_92]], %[[VAL_86]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_92]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_94:.*]]: i32, %[[VAL_95:.*]]: i32, %[[VAL_96:.*]]: i32):
// CHECK:             %[[VAL_97:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : i32
// CHECK:             linalg.yield %[[VAL_97]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_98:.*]] = tensor.collapse_shape %[[VAL_93]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_99:.*]] = arith.index_cast %[[VAL_25]] : i32 to index
// CHECK:           %[[VAL_100:.*]] = arith.minsi %[[VAL_99]], %[[VAL_22]] : index
// CHECK:           %[[VAL_101:.*]] = arith.maxsi %[[VAL_100]], %[[VAL_21]] : index
// CHECK:           %[[VAL_102:.*]] = arith.minsi %[[VAL_101]], %[[VAL_22]] : index
// CHECK:           %[[VAL_103:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           %[[VAL_104:.*]] = arith.minsi %[[VAL_102]], %[[VAL_22]] : index
// CHECK:           scf.for %[[VAL_105:.*]] = %[[VAL_21]] to %[[VAL_104]] step %[[VAL_19]] {
// CHECK:             %[[VAL_106:.*]] = tensor.extract %[[VAL_98]]{{\[}}%[[VAL_105]]] : tensor<16xi32>
// CHECK:             %[[VAL_107:.*]] = arith.index_cast %[[VAL_106]] : i32 to index
// CHECK:             %[[VAL_108:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_107]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_109:.*]] = memref.subview %[[VAL_108]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_110:.*]] = memref.subview %[[VAL_103]]{{\[}}%[[VAL_105]], 0] [1, 8] [1, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_109]], %[[VAL_110]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_111:.*]] = bufferization.to_tensor %[[VAL_103]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_112:.*]] = arith.muli %[[VAL_49]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_113:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_114:.*]] = arith.muli %[[VAL_113]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_115:.*]] = arith.addi %[[VAL_112]], %[[VAL_114]] : i64
// CHECK:           %[[VAL_116:.*]] = arith.index_cast %[[VAL_115]] : i64 to index
// CHECK:           %[[VAL_117:.*]] = linalg.fill ins(%[[VAL_11]] : i64) outs(%[[VAL_52]] : tensor<16x1xi64>) -> tensor<16x1xi64>
// CHECK:           %[[VAL_118:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_53]], %[[VAL_117]] : tensor<16x1xi64>, tensor<16x1xi64>) outs(%[[VAL_53]] : tensor<16x1xi64>) {
// CHECK:           ^bb0(%[[VAL_119:.*]]: i64, %[[VAL_120:.*]]: i64, %[[VAL_121:.*]]: i64):
// CHECK:             %[[VAL_122:.*]] = arith.muli %[[VAL_119]], %[[VAL_120]] : i64
// CHECK:             linalg.yield %[[VAL_122]] : i64
// CHECK:           } -> tensor<16x1xi64>
// CHECK:           %[[VAL_123:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_124:.*]] = arith.index_cast %[[VAL_123]] : index to i32
// CHECK:           %[[VAL_125:.*]] = linalg.fill ins(%[[VAL_124]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_126:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_64]], %[[VAL_125]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_64]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_127:.*]]: i32, %[[VAL_128:.*]]: i32, %[[VAL_129:.*]]: i32):
// CHECK:             %[[VAL_130:.*]] = arith.muli %[[VAL_127]], %[[VAL_128]] : i32
// CHECK:             linalg.yield %[[VAL_130]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_131:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_126]], %[[VAL_125]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_126]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_132:.*]]: i32, %[[VAL_133:.*]]: i32, %[[VAL_134:.*]]: i32):
// CHECK:             %[[VAL_135:.*]] = arith.muli %[[VAL_132]], %[[VAL_133]] : i32
// CHECK:             linalg.yield %[[VAL_135]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_136:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_118]] : tensor<16x1xi64>) outs(%[[VAL_23]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_137:.*]]: i64, %[[VAL_138:.*]]: i32):
// CHECK:             %[[VAL_139:.*]] = arith.trunci %[[VAL_137]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_139]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_140:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_131]], %[[VAL_136]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_131]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_141:.*]]: i32, %[[VAL_142:.*]]: i32, %[[VAL_143:.*]]: i32):
// CHECK:             %[[VAL_144:.*]] = arith.addi %[[VAL_141]], %[[VAL_142]] : i32
// CHECK:             linalg.yield %[[VAL_144]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_145:.*]] = arith.index_cast %[[VAL_116]] : index to i32
// CHECK:           %[[VAL_146:.*]] = linalg.fill ins(%[[VAL_145]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_147:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_146]], %[[VAL_140]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_146]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_148:.*]]: i32, %[[VAL_149:.*]]: i32, %[[VAL_150:.*]]: i32):
// CHECK:             %[[VAL_151:.*]] = arith.addi %[[VAL_148]], %[[VAL_149]] : i32
// CHECK:             linalg.yield %[[VAL_151]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_152:.*]] = tensor.collapse_shape %[[VAL_147]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_153:.*]] = %[[VAL_21]] to %[[VAL_104]] step %[[VAL_19]] {
// CHECK:             %[[VAL_154:.*]] = tensor.extract %[[VAL_152]]{{\[}}%[[VAL_153]]] : tensor<16xi32>
// CHECK:             %[[VAL_155:.*]] = arith.index_cast %[[VAL_154]] : i32 to index
// CHECK:             %[[VAL_156:.*]] = tensor.extract_slice %[[VAL_111]]{{\[}}%[[VAL_153]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_157:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_155]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_158:.*]] = memref.subview %[[VAL_157]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_159:.*]] = memref.cast %[[VAL_158]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[?, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_156]] in writable %[[VAL_159]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[?, ?], offset: ?>>) -> ()
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