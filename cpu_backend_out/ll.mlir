module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @triton__0d1d2de(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %4 = llvm.insertvalue %arg2, %3[0] : !llvm.struct<(i64, ptr)> 
    %5 = llvm.insertvalue %arg3, %4[1] : !llvm.struct<(i64, ptr)> 
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(77 : i32) : i32
    %8 = llvm.mlir.constant(128 : index) : i64
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.mlir.constant(64 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mul %arg8, %9  : i32
    %14 = llvm.sext %13 : i32 to i64
    %15 = llvm.mul %14, %8  : i64
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %19 = llvm.load %18 : !llvm.ptr -> !llvm.ptr
    %20 = llvm.insertvalue %17, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %19, %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %15, %21[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(4 : index) : i64
    %24 = llvm.insertvalue %23, %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %8, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.insertvalue %26, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.insertvalue %28, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mlir.constant(64 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.mlir.constant(256 : index) : i64
    %34 = llvm.mlir.zero : !llvm.ptr
    %35 = llvm.getelementptr %34[256] : (!llvm.ptr) -> !llvm.ptr, i32
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.call @malloc(%36) : (i64) -> !llvm.ptr
    %38 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %30, %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %31, %43[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %31, %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %32, %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%12 : i64)
  ^bb1(%47: i64):  // 2 preds: ^bb0, ^bb5
    %48 = llvm.icmp "slt" %47, %6 : i64
    llvm.cond_br %48, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%12 : i64)
  ^bb3(%49: i64):  // 2 preds: ^bb2, ^bb4
    %50 = llvm.icmp "slt" %49, %10 : i64
    llvm.cond_br %50, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %51 = llvm.mlir.constant(64 : index) : i64
    %52 = llvm.mul %47, %51  : i64
    %53 = llvm.add %52, %49  : i64
    %54 = llvm.getelementptr %37[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %7, %54 : i32, !llvm.ptr
    %55 = llvm.add %49, %11  : i64
    llvm.br ^bb3(%55 : i64)
  ^bb5:  // pred: ^bb3
    %56 = llvm.add %47, %11  : i64
    llvm.br ^bb1(%56 : i64)
  ^bb6:  // pred: ^bb1
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %58 = llvm.insertvalue %17, %57[0] : !llvm.struct<(ptr, ptr, i64)> 
    %59 = llvm.insertvalue %19, %58[1] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.insertvalue %17, %62[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %19, %63[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %15, %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(4 : index) : i64
    %67 = llvm.insertvalue %66, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %8, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.mlir.constant(32 : index) : i64
    %70 = llvm.insertvalue %69, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(1 : index) : i64
    %72 = llvm.insertvalue %71, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %74 = llvm.insertvalue %37, %73[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %37, %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mlir.constant(0 : index) : i64
    %77 = llvm.insertvalue %76, %75[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mlir.constant(4 : index) : i64
    %79 = llvm.insertvalue %78, %77[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(64 : index) : i64
    %81 = llvm.insertvalue %80, %79[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(32 : index) : i64
    %83 = llvm.insertvalue %82, %81[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.insertvalue %84, %83[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.intr.stacksave : !llvm.ptr
    %87 = llvm.mlir.constant(2 : i64) : i64
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.alloca %88 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %72, %89 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %90 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %91 = llvm.insertvalue %87, %90[0] : !llvm.struct<(i64, ptr)> 
    %92 = llvm.insertvalue %89, %91[1] : !llvm.struct<(i64, ptr)> 
    %93 = llvm.mlir.constant(2 : i64) : i64
    %94 = llvm.mlir.constant(1 : index) : i64
    %95 = llvm.alloca %94 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %85, %95 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %96 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %97 = llvm.insertvalue %93, %96[0] : !llvm.struct<(i64, ptr)> 
    %98 = llvm.insertvalue %95, %97[1] : !llvm.struct<(i64, ptr)> 
    %99 = llvm.mlir.constant(1 : index) : i64
    %100 = llvm.alloca %99 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %92, %100 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %101 = llvm.alloca %99 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %98, %101 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %102 = llvm.mlir.zero : !llvm.ptr
    %103 = llvm.getelementptr %102[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    llvm.call @memrefCopy(%104, %100, %101) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %86 : !llvm.ptr
    %105 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %106 = llvm.load %arg3 : !llvm.ptr -> !llvm.ptr
    %107 = llvm.getelementptr %arg3[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %108 = llvm.load %107 : !llvm.ptr -> !llvm.ptr
    %109 = llvm.insertvalue %106, %105[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %108, %109[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.mlir.constant(0 : index) : i64
    %112 = llvm.insertvalue %111, %110[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.mlir.constant(4 : index) : i64
    %114 = llvm.insertvalue %113, %112[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.insertvalue %115, %114[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.mlir.constant(64 : index) : i64
    %118 = llvm.insertvalue %117, %116[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.insertvalue %6, %118[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.intr.stacksave : !llvm.ptr
    %121 = llvm.mlir.constant(2 : i64) : i64
    %122 = llvm.mlir.constant(1 : index) : i64
    %123 = llvm.alloca %122 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %46, %123 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %125 = llvm.insertvalue %121, %124[0] : !llvm.struct<(i64, ptr)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(i64, ptr)> 
    %127 = llvm.mlir.constant(2 : i64) : i64
    %128 = llvm.mlir.constant(1 : index) : i64
    %129 = llvm.alloca %128 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %119, %129 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %130 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %131 = llvm.insertvalue %127, %130[0] : !llvm.struct<(i64, ptr)> 
    %132 = llvm.insertvalue %129, %131[1] : !llvm.struct<(i64, ptr)> 
    %133 = llvm.mlir.constant(1 : index) : i64
    %134 = llvm.alloca %133 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %126, %134 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %135 = llvm.alloca %133 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %132, %135 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %136 = llvm.mlir.zero : !llvm.ptr
    %137 = llvm.getelementptr %136[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %138 = llvm.ptrtoint %137 : !llvm.ptr to i64
    llvm.call @memrefCopy(%138, %134, %135) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %120 : !llvm.ptr
    llvm.return
  }
}

