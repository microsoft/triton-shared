// RUN: triton-shared-opt --tptr-to-llvm %s | FileCheck %s

module {
  func.func @bitcast_ptr_as_src(%arg0: memref<*xi32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = tptr.type_offset i64  : i32
    %1 = tptr.type_offset i32  : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cast = memref.cast %arg1 : memref<*xi32> to memref<1xi32>
    %2 = tptr.from_memref %cast : memref<1xi32> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg0 : memref<*xi32> to memref<1xi32>
    %3 = tptr.from_memref %cast_0 : memref<1xi32> to <#tptr.default_memory_space>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xi32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c16 : index
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    memref.store %c2_i32, %alloc_1[%4] : memref<16xi32>
    %6 = arith.addi %4, %c1 : index
    cf.br ^bb1(%6 : index)
  ^bb3:  // pred: ^bb1
    %7 = arith.muli %c1_i32, %1 : i32
    %8 = tptr.ptradd %3 %7 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    cf.br ^bb4(%c0 : index)
  ^bb4(%9: index):  // 2 preds: ^bb3, ^bb5
    %10 = arith.cmpi slt, %9, %c16 : index
    cf.cond_br %10, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %11 = arith.index_cast %9 : index to i32
    memref.store %11, %alloc[%9] : memref<16xi32>
    %12 = arith.addi %9, %c1 : index
    cf.br ^bb4(%12 : index)
  ^bb6:  // pred: ^bb4
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    cf.br ^bb7(%c0 : index)
  ^bb7(%13: index):  // 2 preds: ^bb6, ^bb8
    %14 = arith.cmpi slt, %13, %c16 : index
    cf.cond_br %14, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    memref.store %8, %alloc_2[%13] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %15 = arith.addi %13, %c1 : index
    cf.br ^bb7(%15 : index)
  ^bb9:  // pred: ^bb7
    cf.br ^bb10(%c0 : index)
  ^bb10(%16: index):  // 2 preds: ^bb9, ^bb11
    %17 = arith.cmpi slt, %16, %c16 : index
    cf.cond_br %17, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %18 = memref.load %alloc_2[%16] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %19 = memref.load %alloc[%16] : memref<16xi32>
    %20 = arith.muli %19, %0 : i32
    %21 = tptr.ptradd %18 %20 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %21, %alloc_2[%16] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %22 = arith.addi %16, %c1 : index
    cf.br ^bb10(%22 : index)
  ^bb12:  // pred: ^bb10
    cf.br ^bb13(%c0 : index)
  ^bb13(%23: index):  // 2 preds: ^bb12, ^bb14
    %24 = arith.cmpi slt, %23, %c16 : index
    cf.cond_br %24, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %25 = memref.load %alloc_2[%23] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %26 = memref.load %alloc_1[%23] : memref<16xi32>
    %27 = arith.muli %26, %0 : i32
    %28 = tptr.ptradd %25 %27 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %28, %alloc_2[%23] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %29 = arith.addi %23, %c1 : index
    cf.br ^bb13(%29 : index)
  ^bb15:  // pred: ^bb13
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16xi64>
    cf.br ^bb16(%c0 : index)
  ^bb16(%30: index):  // 2 preds: ^bb15, ^bb17
    %31 = arith.cmpi slt, %30, %c16 : index
    cf.cond_br %31, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %32 = memref.load %alloc_2[%30] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %33 = tptr.to_memref %32 : <#tptr.default_memory_space> to memref<1xi64>
    %34 = memref.load %33[%c0] : memref<1xi64>
    memref.store %34, %alloc_3[%30] : memref<16xi64>
    %35 = arith.addi %30, %c1 : index
    cf.br ^bb16(%35 : index)
  ^bb18:  // pred: ^bb16
    %36 = tptr.ptradd %2 %7 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    cf.br ^bb19(%c0 : index)
  ^bb19(%37: index):  // 2 preds: ^bb18, ^bb20
    %38 = arith.cmpi slt, %37, %c16 : index
    cf.cond_br %38, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    memref.store %36, %alloc_2[%37] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %39 = arith.addi %37, %c1 : index
    cf.br ^bb19(%39 : index)
  ^bb21:  // pred: ^bb19
    cf.br ^bb22(%c0 : index)
  ^bb22(%40: index):  // 2 preds: ^bb21, ^bb23
    %41 = arith.cmpi slt, %40, %c16 : index
    cf.cond_br %41, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %42 = memref.load %alloc_2[%40] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %43 = memref.load %alloc[%40] : memref<16xi32>
    %44 = arith.muli %43, %0 : i32
    %45 = tptr.ptradd %42 %44 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %45, %alloc_2[%40] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %46 = arith.addi %40, %c1 : index
    cf.br ^bb22(%46 : index)
  ^bb24:  // pred: ^bb22
    cf.br ^bb25(%c0 : index)
  ^bb25(%47: index):  // 2 preds: ^bb24, ^bb26
    %48 = arith.cmpi slt, %47, %c16 : index
    cf.cond_br %48, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %49 = memref.load %alloc_2[%47] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %50 = memref.load %alloc_1[%47] : memref<16xi32>
    %51 = arith.muli %50, %0 : i32
    %52 = tptr.ptradd %49 %51 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %52, %alloc_2[%47] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %53 = arith.addi %47, %c1 : index
    cf.br ^bb25(%53 : index)
  ^bb27:  // pred: ^bb25
    cf.br ^bb28(%c0 : index)
  ^bb28(%54: index):  // 2 preds: ^bb27, ^bb29
    %55 = arith.cmpi slt, %54, %c16 : index
    cf.cond_br %55, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %56 = memref.load %alloc_2[%54] : memref<16x!ptr.ptr<#tptr.default_memory_space>>
    %57 = memref.load %alloc_3[%54] : memref<16xi64>
    %58 = tptr.to_memref %56 : <#tptr.default_memory_space> to memref<1xi64>
    memref.store %57, %58[%c0] : memref<1xi64>
    %59 = arith.addi %54, %c1 : index
    cf.br ^bb28(%59 : index)
  ^bb30:  // pred: ^bb28
    return
  }
}

// CHECK-NOT: tptr.
// CHECK-NOT: ptr.ptr