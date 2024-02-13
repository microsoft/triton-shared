; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @triton__0d1d2de(i64 %0, ptr %1, i64 %2, ptr %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) {
  %12 = insertvalue { i64, ptr } undef, i64 %0, 0
  %13 = insertvalue { i64, ptr } %12, ptr %1, 1
  %14 = insertvalue { i64, ptr } undef, i64 %2, 0
  %15 = insertvalue { i64, ptr } %14, ptr %3, 1
  %16 = mul i32 %8, 4
  %17 = sext i32 %16 to i64
  %18 = mul i64 %17, 128
  %19 = load ptr, ptr %1, align 8
  %20 = getelementptr ptr, ptr %1, i32 1
  %21 = load ptr, ptr %20, align 8
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %19, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %21, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %18, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 4, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 128, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 64, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 1, 4, 1
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 256) to i64))
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 0, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 4, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 64, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 64, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 1, 4, 1
  br label %37

37:                                               ; preds = %49, %11
  %38 = phi i64 [ %50, %49 ], [ 0, %11 ]
  %39 = icmp slt i64 %38, 4
  br i1 %39, label %40, label %51

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %44, %40
  %42 = phi i64 [ %48, %44 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 64
  br i1 %43, label %44, label %49

44:                                               ; preds = %41
  %45 = mul i64 %38, 64
  %46 = add i64 %45, %42
  %47 = getelementptr i32, ptr %29, i64 %46
  store i32 77, ptr %47, align 4
  %48 = add i64 %42, 1
  br label %41

49:                                               ; preds = %41
  %50 = add i64 %38, 1
  br label %37

51:                                               ; preds = %37
  %52 = insertvalue { ptr, ptr, i64 } undef, ptr %19, 0
  %53 = insertvalue { ptr, ptr, i64 } %52, ptr %21, 1
  %54 = insertvalue { ptr, ptr, i64 } %53, i64 0, 2
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %19, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, ptr %21, 1
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 %18, 2
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 4, 3, 0
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 128, 4, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 32, 3, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 1, 4, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, ptr %29, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 0, 2
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 4, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 64, 4, 0
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 32, 3, 1
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 1, 4, 1
  %69 = call ptr @llvm.stacksave.p0()
  %70 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, ptr %70, align 8
  %71 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %70, 1
  %72 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, ptr %72, align 8
  %73 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %72, 1
  %74 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %71, ptr %74, align 8
  %75 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %73, ptr %75, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), ptr %74, ptr %75)
  call void @llvm.stackrestore.p0(ptr %69)
  %76 = load ptr, ptr %3, align 8
  %77 = getelementptr ptr, ptr %3, i32 1
  %78 = load ptr, ptr %77, align 8
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %76, 0
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, ptr %78, 1
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, i64 0, 2
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 4, 3, 0
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 1, 4, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 64, 3, 1
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 4, 4, 1
  %86 = call ptr @llvm.stacksave.p0()
  %87 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %87, align 8
  %88 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %87, 1
  %89 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, ptr %89, align 8
  %90 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %89, 1
  %91 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %88, ptr %91, align 8
  %92 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %90, ptr %92, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), ptr %91, ptr %92)
  call void @llvm.stackrestore.p0(ptr %86)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
