// Test all triton ops on tensor of pointers are converted to linalg.generic
// and all triton ops on pointers are converted to ops in the pointer dialect.
// Original triton program:
//    @triton.jit
//    def tensor_ptr(in_ptr0, out_ptr0):
//        ints = tl.load(in_ptr0 + tl.arange(0, 16)).to(tl.int64)
//        ptrs = ints.to(tl.pointer_type(tl.int32))
//        vals = tl.load(ptrs)
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        ints_2 = out_ptrs.to(tl.int64) + vals
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        out_ptrs_i64 = out_ptrs.to(tl.pointer_type(tl.int64))
//        out_ptrs_i64 += 2
//        out_ptrs_i32 = out_ptrs_i64.to(tl.pointer_type(tl.int32))
//        tl.store(out_ptrs_i32, ints_2.to(tl.int32))

// RUN: triton-shared-opt  --triton-arith-to-linalg="tensor-ptr-to-linalg"  --triton-to-ptr --canonicalize --cse %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>>
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %9 = tt.ptr_to_int %8 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %10 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64>
    %11 = arith.addi %9, %10 : tensor<16xi64>
    %12 = tt.bitcast %8 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %13 = tt.addptr %12, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %14 = tt.bitcast %13 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %15 = arith.trunci %11 : tensor<16xi64> to tensor<16xi32>
    tt.store %14, %15 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @tensor_ptr(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<i32>, %[[VAL_1:.*]]: !tt.ptr<i32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_9]] : i32) outs(%[[VAL_10]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[VAL_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_10]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32):
// CHECK:             %[[VAL_14:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : index to i32
// CHECK:             linalg.yield %[[VAL_15]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_17:.*]] = linalg.fill ins(%[[VAL_0]] : !tt.ptr<i32>) outs(%[[VAL_16]] : tensor<16x!tt.ptr<i32>>) -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_17]], %[[VAL_12]] : tensor<16x!tt.ptr<i32>>, tensor<16xi32>) outs(%[[VAL_17]] : tensor<16x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: !tt.ptr<i32>, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: !tt.ptr<i32>):
// CHECK:             %[[VAL_22:.*]] = tptr.ptradd %[[VAL_19]] %[[VAL_20]] : !tt.ptr<i32>, i32 to !tt.ptr<i32>
// CHECK:             linalg.yield %[[VAL_22]] : !tt.ptr<i32>
// CHECK:           } -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_18]] : tensor<16x!tt.ptr<i32>>) outs(%[[VAL_10]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: !tt.ptr<i32>, %[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = tptr.to_memref %[[VAL_24]] : !tt.ptr<i32> to memref<1xi32>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_8]]] : memref<1xi32>
// CHECK:             linalg.yield %[[VAL_27]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_28:.*]] = tensor.empty() : tensor<16xi64>
// CHECK:           %[[VAL_29:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_23]] : tensor<16xi32>) outs(%[[VAL_28]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_30:.*]]: i32, %[[VAL_31:.*]]: i64):
// CHECK:             %[[VAL_32:.*]] = arith.extsi %[[VAL_30]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_32]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[VAL_33:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_29]] : tensor<16xi64>) outs(%[[VAL_16]] : tensor<16x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_34:.*]]: i64, %[[VAL_35:.*]]: !tt.ptr<i32>):
// CHECK:             %[[VAL_36:.*]] = tptr.inttoptr %[[VAL_34]] : i64 to !tt.ptr<i32>
// CHECK:             linalg.yield %[[VAL_36]] : !tt.ptr<i32>
// CHECK:           } -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_37:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_33]] : tensor<16x!tt.ptr<i32>>) outs(%[[VAL_10]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_38:.*]]: !tt.ptr<i32>, %[[VAL_39:.*]]: i32):
// CHECK:             %[[VAL_40:.*]] = tptr.to_memref %[[VAL_38]] : !tt.ptr<i32> to memref<1xi32>
// CHECK:             %[[VAL_41:.*]] = memref.load %[[VAL_40]]{{\[}}%[[VAL_8]]] : memref<1xi32>
// CHECK:             linalg.yield %[[VAL_41]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_42:.*]] = linalg.fill ins(%[[VAL_1]] : !tt.ptr<i32>) outs(%[[VAL_16]] : tensor<16x!tt.ptr<i32>>) -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_43:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_42]], %[[VAL_12]] : tensor<16x!tt.ptr<i32>>, tensor<16xi32>) outs(%[[VAL_42]] : tensor<16x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_44:.*]]: !tt.ptr<i32>, %[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: !tt.ptr<i32>):
// CHECK:             %[[VAL_47:.*]] = tptr.ptradd %[[VAL_44]] %[[VAL_45]] : !tt.ptr<i32>, i32 to !tt.ptr<i32>
// CHECK:             linalg.yield %[[VAL_47]] : !tt.ptr<i32>
// CHECK:           } -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_48:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_43]] : tensor<16x!tt.ptr<i32>>) outs(%[[VAL_28]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_49:.*]]: !tt.ptr<i32>, %[[VAL_50:.*]]: i64):
// CHECK:             %[[VAL_51:.*]] = tptr.ptrtoint %[[VAL_49]] : !tt.ptr<i32> to i64
// CHECK:             linalg.yield %[[VAL_51]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[VAL_52:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_37]] : tensor<16xi32>) outs(%[[VAL_28]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_53:.*]]: i32, %[[VAL_54:.*]]: i64):
// CHECK:             %[[VAL_55:.*]] = arith.extsi %[[VAL_53]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_55]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[VAL_56:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_48]], %[[VAL_52]] : tensor<16xi64>, tensor<16xi64>) outs(%[[VAL_48]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_57:.*]]: i64, %[[VAL_58:.*]]: i64, %[[VAL_59:.*]]: i64):
// CHECK:             %[[VAL_60:.*]] = arith.addi %[[VAL_57]], %[[VAL_58]] : i64
// CHECK:             linalg.yield %[[VAL_60]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[VAL_61:.*]] = tensor.empty() : tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_62:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_43]] : tensor<16x!tt.ptr<i32>>) outs(%[[VAL_61]] : tensor<16x!tt.ptr<i64>>) {
// CHECK:           ^bb0(%[[VAL_63:.*]]: !tt.ptr<i32>, %[[VAL_64:.*]]: !tt.ptr<i64>):
// CHECK:             %[[VAL_65:.*]] = builtin.unrealized_conversion_cast %[[VAL_63]] : !tt.ptr<i32> to !tt.ptr<i64>
// CHECK:             linalg.yield %[[VAL_65]] : !tt.ptr<i64>
// CHECK:           } -> tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_66:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_62]], %[[VAL_11]] : tensor<16x!tt.ptr<i64>>, tensor<16xi32>) outs(%[[VAL_62]] : tensor<16x!tt.ptr<i64>>) {
// CHECK:           ^bb0(%[[VAL_67:.*]]: !tt.ptr<i64>, %[[VAL_68:.*]]: i32, %[[VAL_69:.*]]: !tt.ptr<i64>):
// CHECK:             %[[VAL_70:.*]] = tptr.ptradd %[[VAL_67]] %[[VAL_68]] : !tt.ptr<i64>, i32 to !tt.ptr<i64>
// CHECK:             linalg.yield %[[VAL_70]] : !tt.ptr<i64>
// CHECK:           } -> tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_71:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_66]] : tensor<16x!tt.ptr<i64>>) outs(%[[VAL_16]] : tensor<16x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_72:.*]]: !tt.ptr<i64>, %[[VAL_73:.*]]: !tt.ptr<i32>):
// CHECK:             %[[VAL_74:.*]] = builtin.unrealized_conversion_cast %[[VAL_72]] : !tt.ptr<i64> to !tt.ptr<i32>
// CHECK:             linalg.yield %[[VAL_74]] : !tt.ptr<i32>
// CHECK:           } -> tensor<16x!tt.ptr<i32>>
// CHECK:           %[[VAL_75:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_56]] : tensor<16xi64>) outs(%[[VAL_10]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_76:.*]]: i64, %[[VAL_77:.*]]: i32):
// CHECK:             %[[VAL_78:.*]] = arith.trunci %[[VAL_76]] : i64 to i32
// CHECK:             linalg.yield %[[VAL_78]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_71]], %[[VAL_75]] : tensor<16x!tt.ptr<i32>>, tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_79:.*]]: !tt.ptr<i32>, %[[VAL_80:.*]]: i32):
// CHECK:             %[[VAL_81:.*]] = tptr.to_memref %[[VAL_79]] : !tt.ptr<i32> to memref<1xi32>
// CHECK:             memref.store %[[VAL_80]], %[[VAL_81]]{{\[}}%[[VAL_8]]] : memref<1xi32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }

