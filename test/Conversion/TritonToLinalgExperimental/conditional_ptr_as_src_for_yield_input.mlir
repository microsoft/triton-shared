// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module attributes {} {
  tt.func public @gather_kernel(%arg0: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %2 = arith.muli %0, %arg6 : i32
    %3 = tt.addptr %arg1, %2 : !tt.ptr<i64>, i32
    %4 = tt.splat %3 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %5 = tt.addptr %4, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %7 = arith.cmpi slt, %1, %6 : tensor<512xi32>
    %8 = tt.load %5, %7 : tensor<512x!tt.ptr<i64>>
    %9 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %10 = scf.if %9 -> (tensor<512x!tt.ptr<i64>>) {
      %15 = arith.extsi %arg5 : i32 to i64
      %16 = tt.splat %15 : i64 -> tensor<512xi64>
      %17 = arith.muli %8, %16 : tensor<512xi64>
      %18 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %19 = tt.addptr %18, %17 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      %20 = tt.addptr %19, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
      scf.yield %20 : tensor<512x!tt.ptr<i64>>
    } else {
      %15 = arith.muli %0, %arg5 : i32
      %16 = tt.addptr %arg0, %15 : !tt.ptr<i64>, i32
      %17 = tt.splat %16 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %18 = tt.addptr %17, %8 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      scf.yield %18 : tensor<512x!tt.ptr<i64>>
    }
    %11 = tt.load %10, %7 : tensor<512x!tt.ptr<i64>>
    %12 = tt.addptr %arg2, %2 : !tt.ptr<i64>, i32
    %13 = tt.splat %12 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %14 = tt.addptr %13, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    tt.store %14, %11, %7 : tensor<512x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @gather_kernel(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                             %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                             %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                             %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                             %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                             %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                             %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_14:.*]] = tptr.type_offset i64  : i32
// CHECK:           %[[VAL_15:.*]] = tptr.type_offset i64  : i64
// CHECK:           %[[VAL_16:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_0]] : memref<*xi64> to memref<1xi64>
// CHECK:           %[[VAL_20:.*]] = tptr.from_memref %[[VAL_19]] : memref<1xi64> to <#{{.*}}>
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:           %[[VAL_22:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_21]] : tensor<512xi32>) {
// CHECK:           ^bb0(%[[VAL_23:.*]]: i32):
// CHECK:             %[[VAL_24:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : index to i32
// CHECK:             linalg.yield %[[VAL_25]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : i32 to index
// CHECK:           %[[VAL_28:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_27]]], sizes: [512], strides: [1] : memref<*xi64> to memref<512xi64, strided<[1], offset: ?>>
// CHECK:           %[[VAL_29:.*]] = linalg.fill ins(%[[VAL_3]] : i32) outs(%[[VAL_21]] : tensor<512xi32>) -> tensor<512xi32>
// CHECK:           %[[VAL_30:.*]] = tensor.empty() : tensor<512xi1>
// CHECK:           %[[VAL_31:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_22]], %[[VAL_29]] : tensor<512xi32>, tensor<512xi32>) outs(%[[VAL_30]] : tensor<512xi1>) {
// CHECK:           ^bb0(%[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i1):
// CHECK:             %[[VAL_35:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:             linalg.yield %[[VAL_35]] : i1
// CHECK:           } -> tensor<512xi1>
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = arith.minsi %[[VAL_36]], %[[VAL_16]] : index
// CHECK:           %[[VAL_38:.*]] = arith.maxsi %[[VAL_37]], %[[VAL_17]] : index
// CHECK:           %[[VAL_39:.*]] = memref.alloc() : memref<512xi64>
// CHECK:           %[[VAL_40:.*]] = memref.subview %[[VAL_28]][0] {{\[}}%[[VAL_38]]] [1] : memref<512xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
// CHECK:           %[[VAL_41:.*]] = memref.subview %[[VAL_39]][0] {{\[}}%[[VAL_38]]] [1] : memref<512xi64> to memref<?xi64, strided<[1]>>
// CHECK:           memref.copy %[[VAL_40]], %[[VAL_41]] : memref<?xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1]>>
// CHECK:           %[[VAL_42:.*]] = bufferization.to_tensor %[[VAL_39]] restrict writable : memref<512xi64> to tensor<512xi64>
// CHECK:           %[[VAL_43:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_44:.*]] = scf.if %[[VAL_43]] -> (memref<512xi64, strided<[?], offset: ?>>) {
// CHECK:             %[[VAL_45:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK:             %[[VAL_46:.*]] = tensor.empty() : tensor<512xi64>
// CHECK:             %[[VAL_47:.*]] = linalg.fill ins(%[[VAL_45]] : i64) outs(%[[VAL_46]] : tensor<512xi64>) -> tensor<512xi64>
// CHECK:             %[[VAL_48:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_42]], %[[VAL_47]] : tensor<512xi64>, tensor<512xi64>) outs(%[[VAL_42]] : tensor<512xi64>) {
// CHECK:             ^bb0(%[[VAL_49:.*]]: i64, %[[VAL_50:.*]]: i64, %[[VAL_51:.*]]: i64):
// CHECK:               %[[VAL_52:.*]] = arith.muli %[[VAL_49]], %[[VAL_50]] : i64
// CHECK:               linalg.yield %[[VAL_52]] : i64
// CHECK:             } -> tensor<512xi64>
// CHECK:             %[[VAL_53:.*]] = tensor.empty() : tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_54:.*]] = linalg.fill ins(%[[VAL_20]] : !ptr.ptr<#{{.*}}>) outs(%[[VAL_53]] : tensor<512x!ptr.ptr<#{{.*}}>>) -> tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_55:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_54]], %[[VAL_48]] : tensor<512x!ptr.ptr<#{{.*}}>>, tensor<512xi64>) outs(%[[VAL_54]] : tensor<512x!ptr.ptr<#{{.*}}>>) {
// CHECK:             ^bb0(%[[VAL_56:.*]]: !ptr.ptr<#{{.*}}>, %[[VAL_57:.*]]: i64, %[[VAL_58:.*]]: !ptr.ptr<#{{.*}}>):
// CHECK:               %[[VAL_59:.*]] = arith.muli %[[VAL_57]], %[[VAL_15]] : i64
// CHECK:               %[[VAL_60:.*]] = tptr.ptradd %[[VAL_56]] %[[VAL_59]] : <#{{.*}}>, i64 to <#{{.*}}>
// CHECK:               linalg.yield %[[VAL_60]] : !ptr.ptr<#{{.*}}>
// CHECK:             } -> tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_61:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_55]], %[[VAL_22]] : tensor<512x!ptr.ptr<#{{.*}}>>, tensor<512xi32>) outs(%[[VAL_55]] : tensor<512x!ptr.ptr<#{{.*}}>>) {
// CHECK:             ^bb0(%[[VAL_62:.*]]: !ptr.ptr<#{{.*}}>, %[[VAL_63:.*]]: i32, %[[VAL_64:.*]]: !ptr.ptr<#{{.*}}>):
// CHECK:               %[[VAL_65:.*]] = arith.muli %[[VAL_63]], %[[VAL_14]] : i32
// CHECK:               %[[VAL_66:.*]] = tptr.ptradd %[[VAL_62]] %[[VAL_65]] : <#{{.*}}>, i32 to <#{{.*}}>
// CHECK:               linalg.yield %[[VAL_66]] : !ptr.ptr<#{{.*}}>
// CHECK:             } -> tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_67:.*]] = builtin.unrealized_conversion_cast %[[VAL_61]] : tensor<512x!ptr.ptr<#{{.*}}>> to tensor<512x!tt.ptr<i64>>
// CHECK:             %[[VAL_68:.*]], %[[VAL_69:.*]], %[[VAL_70:.*]] = "tts.get_structured_state"(%[[VAL_67]]) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<512x!tt.ptr<i64>>) -> (tensor<512x!tt.ptr<i64>>, index, index)
// CHECK:             %[[VAL_71:.*]] = builtin.unrealized_conversion_cast %[[VAL_68]] : tensor<512x!tt.ptr<i64>> to memref<512xi64, strided<[?], offset: ?>>
// CHECK:             scf.yield %[[VAL_71]] : memref<512xi64, strided<[?], offset: ?>>
// CHECK:           } else {
// CHECK:             %[[VAL_72:.*]] = arith.muli %[[VAL_10]], %[[VAL_5]] : i32
// CHECK:             %[[VAL_73:.*]] = arith.muli %[[VAL_72]], %[[VAL_14]] : i32
// CHECK:             %[[VAL_74:.*]] = tptr.ptradd %[[VAL_20]] %[[VAL_73]] : <#{{.*}}>, i32 to <#{{.*}}>
// CHECK:             %[[VAL_75:.*]] = tensor.empty() : tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_76:.*]] = linalg.fill ins(%[[VAL_74]] : !ptr.ptr<#{{.*}}>) outs(%[[VAL_75]] : tensor<512x!ptr.ptr<#{{.*}}>>) -> tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_77:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_76]], %[[VAL_42]] : tensor<512x!ptr.ptr<#{{.*}}>>, tensor<512xi64>) outs(%[[VAL_76]] : tensor<512x!ptr.ptr<#{{.*}}>>) {
// CHECK:             ^bb0(%[[VAL_78:.*]]: !ptr.ptr<#{{.*}}>, %[[VAL_79:.*]]: i64, %[[VAL_80:.*]]: !ptr.ptr<#{{.*}}>):
// CHECK:               %[[VAL_81:.*]] = arith.muli %[[VAL_79]], %[[VAL_15]] : i64
// CHECK:               %[[VAL_82:.*]] = tptr.ptradd %[[VAL_78]] %[[VAL_81]] : <#{{.*}}>, i64 to <#{{.*}}>
// CHECK:               linalg.yield %[[VAL_82]] : !ptr.ptr<#{{.*}}>
// CHECK:             } -> tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:             %[[VAL_83:.*]] = builtin.unrealized_conversion_cast %[[VAL_77]] : tensor<512x!ptr.ptr<#{{.*}}>> to tensor<512x!tt.ptr<i64>>
// CHECK:             %[[VAL_84:.*]], %[[VAL_85:.*]], %[[VAL_86:.*]] = "tts.get_structured_state"(%[[VAL_83]]) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<512x!tt.ptr<i64>>) -> (tensor<512x!tt.ptr<i64>>, index, index)
// CHECK:             %[[VAL_87:.*]] = builtin.unrealized_conversion_cast %[[VAL_84]] : tensor<512x!tt.ptr<i64>> to memref<512xi64, strided<[?], offset: ?>>
// CHECK:             scf.yield %[[VAL_87]] : memref<512xi64, strided<[?], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_88:.*]] = builtin.unrealized_conversion_cast %[[VAL_44]] : memref<512xi64, strided<[?], offset: ?>> to tensor<512x!ptr.ptr<#{{.*}}>>
// CHECK:           %[[VAL_89:.*]] = tensor.empty() : tensor<512xi64>
// CHECK:           %[[VAL_90:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_88]], %[[VAL_31]] : tensor<512x!ptr.ptr<#{{.*}}>>, tensor<512xi1>) outs(%[[VAL_89]] : tensor<512xi64>) {
// CHECK:           ^bb0(%[[VAL_91:.*]]: !ptr.ptr<#{{.*}}>, %[[VAL_92:.*]]: i1, %[[VAL_93:.*]]: i64):
// CHECK:             %[[VAL_94:.*]] = tptr.to_memref %[[VAL_91]] : <#{{.*}}> to memref<1xi64>
// CHECK:             %[[VAL_95:.*]] = scf.if %[[VAL_92]] -> (i64) {
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_94]]{{\[}}%[[VAL_17]]] : memref<1xi64>
// CHECK:               scf.yield %[[VAL_96]] : i64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_13]] : i64
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_95]] : i64
// CHECK:           } -> tensor<512xi64>
// CHECK:           %[[VAL_97:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_27]]], sizes: [512], strides: [1] : memref<*xi64> to memref<512xi64, strided<[1], offset: ?>>
// CHECK:           %[[VAL_98:.*]] = tensor.extract_slice %[[VAL_90]][0] {{\[}}%[[VAL_38]]] [1] : tensor<512xi64> to tensor<?xi64>
// CHECK:           %[[VAL_99:.*]] = memref.subview %[[VAL_97]][0] {{\[}}%[[VAL_38]]] [1] : memref<512xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_98]] in writable %[[VAL_99]] : (tensor<?xi64>, memref<?xi64, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
