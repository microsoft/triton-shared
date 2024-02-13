module {
  func.func @reduce_kernel_2d_0d1d2de3de(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !llvm.struct<(i64, ptr)>
    %1 = arith.index_cast %arg7 : i32 to index
    %2 = builtin.unrealized_conversion_cast %1 : index to i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)> 
    %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %6 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %7 = llvm.load %6 : !llvm.ptr -> !llvm.ptr
    %8 = llvm.insertvalue %5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %7, %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %2, %9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.insertvalue %13, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>
    %16 = arith.sitofp %arg7 : i32 to f32
    affine.store %16, %15[0] : memref<1xf32>
    return
  }
}

