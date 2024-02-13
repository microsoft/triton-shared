; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @reduce_kernel_2d_0d(i64 %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7) {
  %9 = insertvalue { i64, ptr } undef, i64 %0, 0
  %10 = insertvalue { i64, ptr } %9, ptr %1, 1
  %11 = load ptr, ptr %1, align 8
  %12 = getelementptr ptr, ptr %1, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %11, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 0, 2
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, i64 1, 3, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 1, 4, 0
  %19 = sext i32 %5 to i64
  %20 = insertvalue { ptr, ptr, i64 } undef, ptr %11, 0
  %21 = insertvalue { ptr, ptr, i64 } %20, ptr %13, 1
  %22 = insertvalue { ptr, ptr, i64 } %21, i64 0, 2
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %11, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, ptr %13, 1
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %19, 2
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 1, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 1, 4, 0
  br label %28

28:                                               ; preds = %57, %8
  %29 = phi i32 [ %58, %57 ], [ 0, %8 ]
  %30 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %36, %57 ], [ %27, %8 ]
  %31 = icmp slt i32 %29, 8
  br i1 %31, label %32, label %59

32:                                               ; preds = %28
  %33 = sitofp i32 %29 to float
  br label %34

34:                                               ; preds = %38, %32
  %35 = phi i32 [ %56, %38 ], [ 0, %32 ]
  %36 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %55, %38 ], [ %30, %32 ]
  %37 = icmp slt i32 %35, 2
  br i1 %37, label %38, label %57

38:                                               ; preds = %34
  %39 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %41 = getelementptr float, ptr %39, i64 %40
  store float %33, ptr %41, align 4
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %44 = insertvalue { ptr, ptr, i64 } undef, ptr %42, 0
  %45 = insertvalue { ptr, ptr, i64 } %44, ptr %43, 1
  %46 = insertvalue { ptr, ptr, i64 } %45, i64 0, 2
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %48 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 3, 0
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 4, 0
  %50 = add i64 %47, 1
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %42, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, ptr %43, 1
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 %50, 2
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 1, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 1, 4, 0
  %56 = add i32 %35, 1
  br label %34

57:                                               ; preds = %34
  %58 = add i32 %29, 1
  br label %28

59:                                               ; preds = %28
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
