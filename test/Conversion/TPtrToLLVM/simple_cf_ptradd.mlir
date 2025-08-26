// RUN: triton-shared-opt --tptr-to-llvm %s | FileCheck %s

module {
  func.func @simple_cf_into_structured_load_2(%arg0: memref<*xi64> {tt.divisibility = 16 : i32}, %arg1: memref<*xi64> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = tptr.type_offset f32  : i32
    %c0 = arith.constant 0 : index
    %1 = tptr.type_offset i64  : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cast = memref.cast %arg2 : memref<*xf32> to memref<1xf32>
    %2 = tptr.from_memref %cast : memref<1xf32> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg1 : memref<*xi64> to memref<1xi64>
    %3 = tptr.from_memref %cast_0 : memref<1xi64> to <#tptr.default_memory_space>
    %cast_1 = memref.cast %arg0 : memref<*xi64> to memref<1xi64>
    %4 = tptr.from_memref %cast_1 : memref<1xi64> to <#tptr.default_memory_space>
    %5 = arith.remsi %arg6, %c2_i32 : i32
    %6 = arith.cmpi eq, %5, %c0_i32 : i32
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = arith.muli %c2_i32, %1 : i32
    %8 = tptr.ptradd %4 %7 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    cf.br ^bb3(%8 : !ptr.ptr<#tptr.default_memory_space>)
  ^bb2:  // pred: ^bb0
    %9 = arith.muli %c3_i32, %1 : i32
    %10 = tptr.ptradd %3 %9 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    cf.br ^bb3(%10 : !ptr.ptr<#tptr.default_memory_space>)
  ^bb3(%11: !ptr.ptr<#tptr.default_memory_space>):  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %12 = arith.muli %arg6, %1 : i32
    %13 = tptr.ptradd %11 %12 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    %14 = tptr.to_memref %13 : <#tptr.default_memory_space> to memref<1xi64>
    %15 = memref.load %14[%c0] : memref<1xi64>
    %16 = arith.muli %arg6, %0 : i32
    %17 = tptr.ptradd %2 %16 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    %18 = arith.sitofp %15 : i64 to f32
    %19 = tptr.to_memref %17 : <#tptr.default_memory_space> to memref<1xf32>
    memref.store %18, %19[%c0] : memref<1xf32>
    return
  }
}


// CHECK-LABEL:   func.func @simple_cf_into_structured_load_2(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<*xi64> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xi64> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) {
// CHECK-DAG:       %[[VAL_9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_10:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       %[[VAL_11:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       %[[VAL_12:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-DAG:       %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_14:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-DAG:       %[[VAL_15:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_2]] : memref<*xf32> to memref<1xf32>
// CHECK:           %[[VAL_18:.*]] = builtin.unrealized_conversion_cast %[[VAL_17]] : memref<1xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_19:.*]] = llvm.extractvalue %[[VAL_18]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_20:.*]] = memref.cast %[[VAL_1]] : memref<*xi64> to memref<1xi64>
// CHECK:           %[[VAL_21:.*]] = builtin.unrealized_conversion_cast %[[VAL_20]] : memref<1xi64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_22:.*]] = llvm.extractvalue %[[VAL_21]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_23:.*]] = memref.cast %[[VAL_0]] : memref<*xi64> to memref<1xi64>
// CHECK:           %[[VAL_24:.*]] = builtin.unrealized_conversion_cast %[[VAL_23]] : memref<1xi64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_25:.*]] = llvm.extractvalue %[[VAL_24]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_26:.*]] = arith.remsi %[[VAL_6]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_26]], %[[VAL_16]] : i32
// CHECK:           cf.cond_br %[[VAL_27]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_28:.*]] = llvm.getelementptr %[[VAL_25]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:           cf.br ^bb3(%[[VAL_28]] : !llvm.ptr)
// CHECK:         ^bb2:
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_22]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:           cf.br ^bb3(%[[VAL_29]] : !llvm.ptr)
// CHECK:         ^bb3(%[[VAL_30:.*]]: !llvm.ptr):
// CHECK:           %[[VAL_31:.*]] = arith.muli %[[VAL_6]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_32:.*]] = llvm.getelementptr %[[VAL_30]]{{\[}}%[[VAL_31]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:           %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_11]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_33]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_34]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_36:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_35]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_36]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_38:.*]] = builtin.unrealized_conversion_cast %[[VAL_37]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi64>
// CHECK:           %[[VAL_39:.*]] = memref.load %[[VAL_38]]{{\[}}%[[VAL_13]]] : memref<1xi64>
// CHECK:           %[[VAL_40:.*]] = arith.muli %[[VAL_6]], %[[VAL_12]] : i32
// CHECK:           %[[VAL_41:.*]] = llvm.getelementptr %[[VAL_19]]{{\[}}%[[VAL_40]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:           %[[VAL_42:.*]] = arith.sitofp %[[VAL_39]] : i64 to f32
// CHECK:           %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_11]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_44:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_43]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_44]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_46:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_45]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_47:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_46]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_48:.*]] = builtin.unrealized_conversion_cast %[[VAL_47]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>
// CHECK:           memref.store %[[VAL_42]], %[[VAL_48]]{{\[}}%[[VAL_13]]] : memref<1xf32>
// CHECK:           return
// CHECK:         }


