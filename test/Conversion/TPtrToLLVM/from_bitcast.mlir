// RUN: triton-shared-opt --tptr-to-llvm %s | FileCheck %s

// -----// IR Dump Before TPtrToLLVM (tptr-to-llvm) ('builtin.module' operation) //----- //
#loc2 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:33)
#loc3 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:55)
#loc4 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:77)
#loc5 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:89)
#loc6 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:101)
#loc7 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:113)
#loc8 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:125)
#loc9 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:137)
#loc20 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":15:10)
#loc23 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":18:10)
#loc26 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":25:11)
#loc27 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":26:11)
#loc30 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":32:11)
#loc34 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":39:11)
#loc38 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":46:11)
#loc39 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":47:11)
#loc42 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":53:11)
#loc45 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":59:5)
module {
  func.func @bitcast_ptr_as_src(%arg0: memref<*xi32> loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:33), %arg1: memref<*xi32> loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:55), %arg2: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:77), %arg3: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:89), %arg4: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:101), %arg5: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:113), %arg6: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:125), %arg7: i32 loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:137)) {
    %c1 = arith.constant 1 : index loc(#loc10)
    %c16 = arith.constant 16 : index loc(#loc10)
    %c0 = arith.constant 0 : index loc(#loc10)
    %0 = tptr.type_offset i64  : i32 loc(#loc11)
    %1 = tptr.type_offset i32  : i32 loc(#loc12)
    %c2_i32 = arith.constant 2 : i32 loc(#loc13)
    %c1_i32 = arith.constant 1 : i32 loc(#loc14)
    %cast = memref.cast %arg1 : memref<*xi32> to memref<1xi32> loc(#loc15)
    %2 = tptr.from_memref %cast : memref<1xi32> to <#tptr.default_memory_space> loc(#loc16)
    %cast_0 = memref.cast %arg0 : memref<*xi32> to memref<1xi32> loc(#loc17)
    %3 = tptr.from_memref %cast_0 : memref<1xi32> to <#tptr.default_memory_space> loc(#loc18)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xi32> loc(#loc19)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16xi32> loc(#loc20)
    cf.br ^bb1(%c0 : index) loc(#loc20)
  ^bb1(%4: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":15:10)):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c16 : index loc(#loc20)
    cf.cond_br %5, ^bb2, ^bb3 loc(#loc20)
  ^bb2:  // pred: ^bb1
    memref.store %c2_i32, %alloc_1[%4] : memref<16xi32> loc(#loc20)
    %6 = arith.addi %4, %c1 : index loc(#loc20)
    cf.br ^bb1(%6 : index) loc(#loc20)
  ^bb3:  // pred: ^bb1
    %7 = arith.muli %c1_i32, %1 : i32 loc(#loc21)
    %8 = tptr.ptradd %3 %7 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc22)
    cf.br ^bb4(%c0 : index) loc(#loc23)
  ^bb4(%9: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":18:10)):  // 2 preds: ^bb3, ^bb5
    %10 = arith.cmpi slt, %9, %c16 : index loc(#loc23)
    cf.cond_br %10, ^bb5, ^bb6 loc(#loc23)
  ^bb5:  // pred: ^bb4
    %11 = arith.index_cast %9 : index to i32 loc(#loc24)
    memref.store %11, %alloc[%9] : memref<16xi32> loc(#loc23)
    %12 = arith.addi %9, %c1 : index loc(#loc23)
    cf.br ^bb4(%12 : index) loc(#loc23)
  ^bb6:  // pred: ^bb4
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc25)
    cf.br ^bb7(%c0 : index) loc(#loc26)
  ^bb7(%13: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":25:11)):  // 2 preds: ^bb6, ^bb8
    %14 = arith.cmpi slt, %13, %c16 : index loc(#loc26)
    cf.cond_br %14, ^bb8, ^bb9 loc(#loc26)
  ^bb8:  // pred: ^bb7
    memref.store %8, %alloc_2[%13] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc26)
    %15 = arith.addi %13, %c1 : index loc(#loc26)
    cf.br ^bb7(%15 : index) loc(#loc26)
  ^bb9:  // pred: ^bb7
    cf.br ^bb10(%c0 : index) loc(#loc27)
  ^bb10(%16: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":26:11)):  // 2 preds: ^bb9, ^bb11
    %17 = arith.cmpi slt, %16, %c16 : index loc(#loc27)
    cf.cond_br %17, ^bb11, ^bb12 loc(#loc27)
  ^bb11:  // pred: ^bb10
    %18 = memref.load %alloc_2[%16] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc27)
    %19 = memref.load %alloc[%16] : memref<16xi32> loc(#loc27)
    %20 = arith.muli %19, %0 : i32 loc(#loc28)
    %21 = tptr.ptradd %18 %20 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc29)
    memref.store %21, %alloc_2[%16] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc27)
    %22 = arith.addi %16, %c1 : index loc(#loc27)
    cf.br ^bb10(%22 : index) loc(#loc27)
  ^bb12:  // pred: ^bb10
    cf.br ^bb13(%c0 : index) loc(#loc30)
  ^bb13(%23: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":32:11)):  // 2 preds: ^bb12, ^bb14
    %24 = arith.cmpi slt, %23, %c16 : index loc(#loc30)
    cf.cond_br %24, ^bb14, ^bb15 loc(#loc30)
  ^bb14:  // pred: ^bb13
    %25 = memref.load %alloc_2[%23] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc30)
    %26 = memref.load %alloc_1[%23] : memref<16xi32> loc(#loc30)
    %27 = arith.muli %26, %0 : i32 loc(#loc31)
    %28 = tptr.ptradd %25 %27 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc32)
    memref.store %28, %alloc_2[%23] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc30)
    %29 = arith.addi %23, %c1 : index loc(#loc30)
    cf.br ^bb13(%29 : index) loc(#loc30)
  ^bb15:  // pred: ^bb13
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16xi64> loc(#loc33)
    cf.br ^bb16(%c0 : index) loc(#loc34)
  ^bb16(%30: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":39:11)):  // 2 preds: ^bb15, ^bb17
    %31 = arith.cmpi slt, %30, %c16 : index loc(#loc34)
    cf.cond_br %31, ^bb17, ^bb18 loc(#loc34)
  ^bb17:  // pred: ^bb16
    %32 = memref.load %alloc_2[%30] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc34)
    %33 = tptr.to_memref %32 : <#tptr.default_memory_space> to memref<1xi64> loc(#loc35)
    %34 = memref.load %33[%c0] : memref<1xi64> loc(#loc36)
    memref.store %34, %alloc_3[%30] : memref<16xi64> loc(#loc34)
    %35 = arith.addi %30, %c1 : index loc(#loc34)
    cf.br ^bb16(%35 : index) loc(#loc34)
  ^bb18:  // pred: ^bb16
    %36 = tptr.ptradd %2 %7 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc37)
    cf.br ^bb19(%c0 : index) loc(#loc38)
  ^bb19(%37: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":46:11)):  // 2 preds: ^bb18, ^bb20
    %38 = arith.cmpi slt, %37, %c16 : index loc(#loc38)
    cf.cond_br %38, ^bb20, ^bb21 loc(#loc38)
  ^bb20:  // pred: ^bb19
    memref.store %36, %alloc_2[%37] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc38)
    %39 = arith.addi %37, %c1 : index loc(#loc38)
    cf.br ^bb19(%39 : index) loc(#loc38)
  ^bb21:  // pred: ^bb19
    cf.br ^bb22(%c0 : index) loc(#loc39)
  ^bb22(%40: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":47:11)):  // 2 preds: ^bb21, ^bb23
    %41 = arith.cmpi slt, %40, %c16 : index loc(#loc39)
    cf.cond_br %41, ^bb23, ^bb24 loc(#loc39)
  ^bb23:  // pred: ^bb22
    %42 = memref.load %alloc_2[%40] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc39)
    %43 = memref.load %alloc[%40] : memref<16xi32> loc(#loc39)
    %44 = arith.muli %43, %0 : i32 loc(#loc40)
    %45 = tptr.ptradd %42 %44 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc41)
    memref.store %45, %alloc_2[%40] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc39)
    %46 = arith.addi %40, %c1 : index loc(#loc39)
    cf.br ^bb22(%46 : index) loc(#loc39)
  ^bb24:  // pred: ^bb22
    cf.br ^bb25(%c0 : index) loc(#loc42)
  ^bb25(%47: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":53:11)):  // 2 preds: ^bb24, ^bb26
    %48 = arith.cmpi slt, %47, %c16 : index loc(#loc42)
    cf.cond_br %48, ^bb26, ^bb27 loc(#loc42)
  ^bb26:  // pred: ^bb25
    %49 = memref.load %alloc_2[%47] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc42)
    %50 = memref.load %alloc_1[%47] : memref<16xi32> loc(#loc42)
    %51 = arith.muli %50, %0 : i32 loc(#loc43)
    %52 = tptr.ptradd %49 %51 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc44)
    memref.store %52, %alloc_2[%47] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc42)
    %53 = arith.addi %47, %c1 : index loc(#loc42)
    cf.br ^bb25(%53 : index) loc(#loc42)
  ^bb27:  // pred: ^bb25
    cf.br ^bb28(%c0 : index) loc(#loc45)
  ^bb28(%54: index loc("/tmp/tmpq_mdk7uo/ttshared.mlir":59:5)):  // 2 preds: ^bb27, ^bb29
    %55 = arith.cmpi slt, %54, %c16 : index loc(#loc45)
    cf.cond_br %55, ^bb29, ^bb30 loc(#loc45)
  ^bb29:  // pred: ^bb28
    %56 = memref.load %alloc_2[%54] : memref<16x!ptr.ptr<#tptr.default_memory_space>> loc(#loc45)
    %57 = memref.load %alloc_3[%54] : memref<16xi64> loc(#loc45)
    %58 = tptr.to_memref %56 : <#tptr.default_memory_space> to memref<1xi64> loc(#loc46)
    memref.store %57, %58[%c0] : memref<1xi64> loc(#loc47)
    %59 = arith.addi %54, %c1 : index loc(#loc45)
    cf.br ^bb28(%59 : index) loc(#loc45)
  ^bb30:  // pred: ^bb28
    return loc(#loc48)
  } loc(#loc1)
} loc(#loc)
#loc = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":3:1)
#loc1 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":4:3)
#loc10 = loc(unknown)
#loc11 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":6:10)
#loc12 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":7:10)
#loc13 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":8:15)
#loc14 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":9:15)
#loc15 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":10:13)
#loc16 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":11:10)
#loc17 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":12:15)
#loc18 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":13:10)
#loc19 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":14:10)
#loc21 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":16:10)
#loc22 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":17:10)
#loc24 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":21:13)
#loc25 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":24:10)
#loc28 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":28:13)
#loc29 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":29:13)
#loc31 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":34:13)
#loc32 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":35:13)
#loc33 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":38:11)
#loc35 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":41:13)
#loc36 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":42:13)
#loc37 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":45:11)
#loc40 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":49:13)
#loc41 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":50:13)
#loc43 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":55:13)
#loc44 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":56:13)
#loc46 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":61:13)
#loc47 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":62:7)
#loc48 = loc("/tmp/tmpq_mdk7uo/ttshared.mlir":65:5)



// CHECK-NOT: tptr.
// CHECK-NOT: ptr.ptr