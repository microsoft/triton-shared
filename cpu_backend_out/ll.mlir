module {
  llvm.func @reduce_kernel_2d_0d(%arg0: i64, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(8 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %12 = llvm.load %11 : !llvm.ptr -> !llvm.ptr
    %13 = llvm.insertvalue %10, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %12, %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.insertvalue %17, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.insertvalue %19, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.sext %arg5 : i32 to i64
    %22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %23 = llvm.insertvalue %10, %22[0] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = llvm.insertvalue %12, %23[1] : !llvm.struct<(ptr, ptr, i64)> 
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.insertvalue %25, %24[2] : !llvm.struct<(ptr, ptr, i64)> 
    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %10, %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %12, %28[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %21, %29[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%5, %34 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb1(%35: i32, %36: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb0, ^bb5
    %37 = llvm.icmp "slt" %35, %4 : i32
    llvm.cond_br %37, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %38 = llvm.sitofp %35 : i32 to f32
    llvm.br ^bb3(%5, %36 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb3(%39: i32, %40: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb2, ^bb4
    %41 = llvm.icmp "slt" %39, %3 : i32
    llvm.cond_br %41, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %42 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.getelementptr %42[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %38, %44 : f32, !llvm.ptr
    %45 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %48 = llvm.insertvalue %45, %47[0] : !llvm.struct<(ptr, ptr, i64)> 
    %49 = llvm.insertvalue %46, %48[1] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.mlir.constant(0 : index) : i64
    %51 = llvm.insertvalue %50, %49[2] : !llvm.struct<(ptr, ptr, i64)> 
    %52 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.extractvalue %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.add %52, %7  : i64
    %56 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.insertvalue %45, %56[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %46, %57[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.insertvalue %55, %58[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.insertvalue %60, %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.insertvalue %62, %61[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.add %39, %6  : i32
    llvm.br ^bb3(%64, %63 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb5:  // pred: ^bb3
    %65 = llvm.add %35, %6  : i32
    llvm.br ^bb1(%65, %40 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

