// RUN: triton-shared-opt --tptr-to-llvm %s | FileCheck %s

#loc = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0)
#loc1 = loc(unknown)
#loc4 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:27)
#loc5 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:16)
#loc6 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":167:44)
#loc7 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":108:26)
#loc12 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":110:22)
#loc18 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":119:16)
#loc30 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":164:7)
#loc31 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":173:30)
#loc33 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":172:16)
#loc34 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:44)
#loc36 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:26)
#loc38 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)
#loc39 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:40)
#loc41 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)
#loc42 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":165:59)
#loc43 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":167:31)
#loc44 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":172:26)
#loc45 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":171:16)
#loc46 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:38)
#loc47 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:59)
#loc48 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:50)
#loc51 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":188:28)
module {
  func.func @_fwd_grouped_kernel_stage1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg1: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg2: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg3: f32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg4: memref<*xi32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg5: memref<*xi32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg6: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg7: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg8: memref<*xf32> {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg9: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg10: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg11: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg12: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg13: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg14: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg15: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg16: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg17: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg18: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg19: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg20: i32 {tt.divisibility = 16 : i32} loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg21: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg22: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg23: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg24: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg25: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0), %arg26: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":50:0)) {
    %c32 = arith.constant 32 : index loc(#loc1)
    %c64 = arith.constant 64 : index loc(#loc1)
    %c1 = arith.constant 1 : index loc(#loc1)
    %0 = tptr.type_offset f32  : i32 loc(#loc1)
    %c0 = arith.constant 0 : index loc(#loc1)
    %1 = tptr.type_offset i32  : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c2_i32 = arith.constant 2 : i32 loc(#loc1)
    %c4 = arith.constant 4 : index loc(#loc1)
    %c8 = arith.constant 8 : index loc(#loc1)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %cast = memref.cast %arg5 : memref<*xi32> to memref<1xi32> loc(#loc2)
    %2 = tptr.from_memref %cast : memref<1xi32> to <#tptr.default_memory_space> loc(#loc2)
    %cast_0 = memref.cast %arg4 : memref<*xi32> to memref<1xi32> loc(#loc3)
    %3 = tptr.from_memref %cast_0 : memref<1xi32> to <#tptr.default_memory_space> loc(#loc3)
    %cast_1 = memref.cast %arg1 : memref<*xf32> to memref<1xf32> loc(#loc4)
    %4 = tptr.from_memref %cast_1 : memref<1xf32> to <#tptr.default_memory_space> loc(#loc4)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32> loc(#loc5)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32> loc(#loc1)
    cf.br ^bb1(%c0 : index) loc(#loc1)
  ^bb1(%5: index loc(unknown)):  // 2 preds: ^bb0, ^bb5
    %6 = arith.cmpi slt, %5, %c64 : index loc(#loc1)
    cf.cond_br %6, ^bb2, ^bb6 loc(#loc1)
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index) loc(#loc1)
  ^bb3(%7: index loc(unknown)):  // 2 preds: ^bb2, ^bb4
    %8 = arith.cmpi slt, %7, %c32 : index loc(#loc1)
    cf.cond_br %8, ^bb4, ^bb5 loc(#loc1)
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc_2[%5, %7] : memref<64x32xf32> loc(#loc1)
    %9 = arith.addi %7, %c1 : index loc(#loc1)
    cf.br ^bb3(%9 : index) loc(#loc1)
  ^bb5:  // pred: ^bb3
    %10 = arith.addi %5, %c1 : index loc(#loc1)
    cf.br ^bb1(%10 : index) loc(#loc1)
  ^bb6:  // pred: ^bb1
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc6)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc1)
    cf.br ^bb7(%c0 : index) loc(#loc1)
  ^bb7(%11: index loc(unknown)):  // 2 preds: ^bb6, ^bb8
    %12 = arith.cmpi slt, %11, %c32 : index loc(#loc1)
    cf.cond_br %12, ^bb8, ^bb9 loc(#loc1)
  ^bb8:  // pred: ^bb7
    memref.store %c0_i32, %alloc_4[%11] : memref<32xi32> loc(#loc1)
    %13 = arith.addi %11, %c1 : index loc(#loc1)
    cf.br ^bb7(%13 : index) loc(#loc1)
  ^bb9:  // pred: ^bb7
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc1)
    cf.br ^bb10(%c0 : index) loc(#loc1)
  ^bb10(%14: index loc(unknown)):  // 2 preds: ^bb9, ^bb11
    %15 = arith.cmpi slt, %14, %c32 : index loc(#loc1)
    cf.cond_br %15, ^bb11, ^bb12 loc(#loc1)
  ^bb11:  // pred: ^bb10
    memref.store %c16_i32, %alloc_5[%14] : memref<32xi32> loc(#loc1)
    %16 = arith.addi %14, %c1 : index loc(#loc1)
    cf.br ^bb10(%16 : index) loc(#loc1)
  ^bb12:  // pred: ^bb10
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64xi32> loc(#loc7)
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<64xi32> loc(#loc1)
    cf.br ^bb13(%c0 : index) loc(#loc1)
  ^bb13(%17: index loc(unknown)):  // 2 preds: ^bb12, ^bb14
    %18 = arith.cmpi slt, %17, %c64 : index loc(#loc1)
    cf.cond_br %18, ^bb14, ^bb15 loc(#loc1)
  ^bb14:  // pred: ^bb13
    memref.store %c64_i32, %alloc_7[%17] : memref<64xi32> loc(#loc1)
    %19 = arith.addi %17, %c1 : index loc(#loc1)
    cf.br ^bb13(%19 : index) loc(#loc1)
  ^bb15:  // pred: ^bb13
    %20 = arith.muli %arg25, %c4_i32 : i32 loc(#loc8)
    %21 = arith.index_cast %20 : i32 to index loc(#loc9)
    %22 = arith.addi %arg25, %c1_i32 : i32 loc(#loc10)
    %23 = arith.muli %22, %c4_i32 : i32 loc(#loc11)
    cf.br ^bb16(%c0 : index) loc(#loc7)
  ^bb16(%24: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":108:26)):  // 2 preds: ^bb15, ^bb17
    %25 = arith.cmpi slt, %24, %c64 : index loc(#loc7)
    cf.cond_br %25, ^bb17, ^bb18 loc(#loc7)
  ^bb17:  // pred: ^bb16
    %26 = arith.index_cast %24 : index to i32 loc(#loc7)
    memref.store %26, %alloc_6[%24] : memref<64xi32> loc(#loc7)
    %27 = arith.addi %24, %c1 : index loc(#loc7)
    cf.br ^bb16(%27 : index) loc(#loc7)
  ^bb18:  // pred: ^bb16
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<64xi1> loc(#loc12)
    cf.br ^bb19(%c0 : index) loc(#loc12)
  ^bb19(%28: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":110:22)):  // 2 preds: ^bb18, ^bb20
    %29 = arith.cmpi slt, %28, %c64 : index loc(#loc12)
    cf.cond_br %29, ^bb20, ^bb21 loc(#loc12)
  ^bb20:  // pred: ^bb19
    %30 = memref.load %alloc_6[%28] : memref<64xi32> loc(#loc12)
    %31 = memref.load %alloc_7[%28] : memref<64xi32> loc(#loc12)
    %32 = arith.cmpi slt, %30, %31 : i32 loc(#loc12)
    memref.store %32, %alloc_8[%28] : memref<64xi1> loc(#loc12)
    %33 = arith.addi %28, %c1 : index loc(#loc12)
    cf.br ^bb19(%33 : index) loc(#loc12)
  ^bb21:  // pred: ^bb19
    %34 = arith.muli %arg24, %1 : i32 loc(#loc2)
    %35 = tptr.ptradd %2 %34 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc2)
    %36 = tptr.to_memref %35 : <#tptr.default_memory_space> to memref<1xi32> loc(#loc13)
    %37 = memref.load %36[%c0] : memref<1xi32> loc(#loc13)
    %38 = arith.muli %arg24, %arg10 : i32 loc(#loc14)
    %39 = arith.index_cast %38 : i32 to index loc(#loc15)
    %40 = arith.index_cast %arg11 : i32 to index loc(#loc15)
    %41 = arith.muli %21, %40 : index loc(#loc16)
    %42 = arith.addi %39, %41 : index loc(#loc17)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%42], sizes: [4, 64], strides: [%40, 1] : memref<*xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>> loc(#loc15)
    %43 = arith.addi %21, %c4 : index loc(#loc18)
    %44 = arith.index_cast %23 : i32 to index loc(#loc18)
    %45 = arith.minsi %43, %44 : index loc(#loc18)
    %46 = arith.maxsi %45, %21 : index loc(#loc18)
    %47 = arith.subi %46, %21 : index loc(#loc18)
    %48 = arith.minsi %43, %c8 : index loc(#loc18)
    %49 = arith.maxsi %48, %21 : index loc(#loc18)
    %50 = arith.subi %49, %21 : index loc(#loc18)
    %51 = arith.minsi %47, %50 : index loc(#loc18)
    %52 = arith.minsi %51, %c4 : index loc(#loc18)
    %alloc_9 = memref.alloc() : memref<4x64xf32> loc(#loc18)
    %53 = arith.cmpi slt, %52, %c4 : index loc(#loc18)
    cf.cond_br %53, ^bb22, ^bb29 loc(#loc18)
  ^bb22:  // pred: ^bb21
    cf.br ^bb23(%c0 : index) loc(#loc18)
  ^bb23(%54: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":119:16)):  // 2 preds: ^bb22, ^bb27
    %55 = arith.cmpi slt, %54, %c4 : index loc(#loc18)
    cf.cond_br %55, ^bb24, ^bb28 loc(#loc18)
  ^bb24:  // pred: ^bb23
    cf.br ^bb25(%c0 : index) loc(#loc18)
  ^bb25(%56: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":119:16)):  // 2 preds: ^bb24, ^bb26
    %57 = arith.cmpi slt, %56, %c64 : index loc(#loc18)
    cf.cond_br %57, ^bb26, ^bb27 loc(#loc18)
  ^bb26:  // pred: ^bb25
    memref.store %cst, %alloc_9[%54, %56] : memref<4x64xf32> loc(#loc18)
    %58 = arith.addi %56, %c1 : index loc(#loc18)
    cf.br ^bb25(%58 : index) loc(#loc18)
  ^bb27:  // pred: ^bb25
    %59 = arith.addi %54, %c1 : index loc(#loc18)
    cf.br ^bb23(%59 : index) loc(#loc18)
  ^bb28:  // pred: ^bb23
    cf.br ^bb29 loc(#loc18)
  ^bb29:  // 2 preds: ^bb21, ^bb28
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<4x64xf32, strided<[?, 1], offset: ?>> -> memref<f32>, index, index, index, index, index loc(#loc18)
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [%52, 64], strides: [%strides#0, 1] : memref<f32> to memref<?x64xf32, strided<[?, 1], offset: ?>> loc(#loc18)
    %reinterpret_cast_11 = memref.reinterpret_cast %alloc_9 to offset: [0], sizes: [%52, 64], strides: [64, 1] : memref<4x64xf32> to memref<?x64xf32, strided<[64, 1]>> loc(#loc18)
    memref.copy %reinterpret_cast_10, %reinterpret_cast_11 : memref<?x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1]>> loc(#loc18)
    %60 = arith.muli %arg24, %arg19 : i32 loc(#loc19)
    %61 = arith.index_cast %60 : i32 to index loc(#loc9)
    %62 = arith.index_cast %arg20 : i32 to index loc(#loc9)
    %63 = arith.muli %21, %62 : index loc(#loc20)
    %64 = arith.addi %61, %63 : index loc(#loc21)
    %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%64], sizes: [4, 64], strides: [%62, 1] : memref<*xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>> loc(#loc9)
    %reinterpret_cast_13 = memref.reinterpret_cast %alloc_9 to offset: [0], sizes: [%52, 64], strides: [64, 1] : memref<4x64xf32> to memref<?x64xf32, strided<[64, 1]>> loc(#loc22)
    %base_buffer_14, %offset_15, %sizes_16:2, %strides_17:2 = memref.extract_strided_metadata %reinterpret_cast_12 : memref<4x64xf32, strided<[?, 1], offset: ?>> -> memref<f32>, index, index, index, index, index loc(#loc22)
    %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_14 to offset: [%offset_15], sizes: [%52, 64], strides: [%strides_17#0, 1] : memref<f32> to memref<?x64xf32, strided<[?, 1], offset: ?>> loc(#loc22)
    memref.copy %reinterpret_cast_13, %reinterpret_cast_18 : memref<?x64xf32, strided<[64, 1]>> to memref<?x64xf32, strided<[?, 1], offset: ?>> loc(#loc22)
    %65 = arith.addi %37, %c1_i32 : i32 loc(#loc59)
    %66 = arith.divsi %65, %c2_i32 : i32 loc(#loc60)
    %67 = arith.muli %66, %arg26 : i32 loc(#loc26)
    %68 = arith.addi %67, %66 : i32 loc(#loc27)
    %69 = arith.minsi %68, %37 : i32 loc(#loc28)
    %70 = arith.cmpi sgt, %69, %67 : i32 loc(#loc29)
    cf.cond_br %70, ^bb30, ^bb174 loc(#loc30)
  ^bb30:  // pred: ^bb29
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc6)
    cf.br ^bb31(%c0 : index) loc(#loc6)
  ^bb31(%71: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":167:44)):  // 2 preds: ^bb30, ^bb32
    %72 = arith.cmpi slt, %71, %c32 : index loc(#loc6)
    cf.cond_br %72, ^bb32, ^bb33 loc(#loc6)
  ^bb32:  // pred: ^bb31
    %73 = arith.index_cast %71 : index to i32 loc(#loc6)
    memref.store %73, %alloc_19[%71] : memref<32xi32> loc(#loc6)
    %74 = arith.addi %71, %c1 : index loc(#loc6)
    cf.br ^bb31(%74 : index) loc(#loc6)
  ^bb33:  // pred: ^bb31
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc31)
    cf.br ^bb34(%c0 : index) loc(#loc31)
  ^bb34(%75: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":173:30)):  // 2 preds: ^bb33, ^bb35
    %76 = arith.cmpi slt, %75, %c32 : index loc(#loc31)
    cf.cond_br %76, ^bb35, ^bb36 loc(#loc31)
  ^bb35:  // pred: ^bb34
    memref.store %69, %alloc_20[%75] : memref<32xi32> loc(#loc31)
    %77 = arith.addi %75, %c1 : index loc(#loc31)
    cf.br ^bb34(%77 : index) loc(#loc31)
  ^bb36:  // pred: ^bb34
    %78 = arith.muli %arg9, %arg24 : i32 loc(#loc32)
    %79 = arith.muli %78, %1 : i32 loc(#loc3)
    %80 = tptr.ptradd %3 %79 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc3)
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc33)
    cf.br ^bb37(%c0 : index) loc(#loc33)
  ^bb37(%81: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":172:16)):  // 2 preds: ^bb36, ^bb38
    %82 = arith.cmpi slt, %81, %c32 : index loc(#loc33)
    cf.cond_br %82, ^bb38, ^bb39 loc(#loc33)
  ^bb38:  // pred: ^bb37
    memref.store %80, %alloc_21[%81] : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc33)
    %83 = arith.addi %81, %c1 : index loc(#loc33)
    cf.br ^bb37(%83 : index) loc(#loc33)
  ^bb39:  // pred: ^bb37
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32> loc(#loc34)
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32> loc(#loc34)
    cf.br ^bb40(%c0 : index) loc(#loc34)
  ^bb40(%84: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:44)):  // 2 preds: ^bb39, ^bb44
    %85 = arith.cmpi slt, %84, %c1 : index loc(#loc34)
    cf.cond_br %85, ^bb41, ^bb45 loc(#loc34)
  ^bb41:  // pred: ^bb40
    cf.br ^bb42(%c0 : index) loc(#loc34)
  ^bb42(%86: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:44)):  // 2 preds: ^bb41, ^bb43
    %87 = arith.cmpi slt, %86, %c32 : index loc(#loc34)
    cf.cond_br %87, ^bb43, ^bb44 loc(#loc34)
  ^bb43:  // pred: ^bb42
    memref.store %arg12, %alloc_23[%84, %86] : memref<1x32xi32> loc(#loc34)
    %88 = arith.addi %86, %c1 : index loc(#loc34)
    cf.br ^bb42(%88 : index) loc(#loc34)
  ^bb44:  // pred: ^bb42
    %89 = arith.addi %84, %c1 : index loc(#loc34)
    cf.br ^bb40(%89 : index) loc(#loc34)
  ^bb45:  // pred: ^bb40
    %90 = arith.muli %arg25, %arg13 : i32 loc(#loc35)
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32> loc(#loc36)
    cf.br ^bb46(%c0 : index) loc(#loc36)
  ^bb46(%91: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:26)):  // 2 preds: ^bb45, ^bb50
    %92 = arith.cmpi slt, %91, %c1 : index loc(#loc36)
    cf.cond_br %92, ^bb47, ^bb51 loc(#loc36)
  ^bb47:  // pred: ^bb46
    cf.br ^bb48(%c0 : index) loc(#loc36)
  ^bb48(%93: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:26)):  // 2 preds: ^bb47, ^bb49
    %94 = arith.cmpi slt, %93, %c32 : index loc(#loc36)
    cf.cond_br %94, ^bb49, ^bb50 loc(#loc36)
  ^bb49:  // pred: ^bb48
    memref.store %90, %alloc_24[%91, %93] : memref<1x32xi32> loc(#loc36)
    %95 = arith.addi %93, %c1 : index loc(#loc36)
    cf.br ^bb48(%95 : index) loc(#loc36)
  ^bb50:  // pred: ^bb48
    %96 = arith.addi %91, %c1 : index loc(#loc36)
    cf.br ^bb46(%96 : index) loc(#loc36)
  ^bb51:  // pred: ^bb46
    %reinterpret_cast_25 = memref.reinterpret_cast %alloc_6 to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<64xi32> to memref<64x1xi32> loc(#loc37)
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi32> loc(#loc38)
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi32> loc(#loc38)
    cf.br ^bb52(%c0 : index) loc(#loc38)
  ^bb52(%97: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb51, ^bb56
    %98 = arith.cmpi slt, %97, %c64 : index loc(#loc38)
    cf.cond_br %98, ^bb53, ^bb57 loc(#loc38)
  ^bb53:  // pred: ^bb52
    cf.br ^bb54(%c0 : index) loc(#loc38)
  ^bb54(%99: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb53, ^bb55
    %100 = arith.cmpi slt, %99, %c32 : index loc(#loc38)
    cf.cond_br %100, ^bb55, ^bb56 loc(#loc38)
  ^bb55:  // pred: ^bb54
    %101 = memref.load %reinterpret_cast_25[%97, %c0] : memref<64x1xi32> loc(#loc38)
    memref.store %101, %alloc_27[%97, %99] : memref<64x32xi32> loc(#loc38)
    %102 = arith.addi %99, %c1 : index loc(#loc38)
    cf.br ^bb54(%102 : index) loc(#loc38)
  ^bb56:  // pred: ^bb54
    %103 = arith.addi %97, %c1 : index loc(#loc38)
    cf.br ^bb52(%103 : index) loc(#loc38)
  ^bb57:  // pred: ^bb52
    cf.br ^bb58(%c0 : index) loc(#loc39)
  ^bb58(%104: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:40)):  // 2 preds: ^bb57, ^bb62
    %105 = arith.cmpi slt, %104, %c1 : index loc(#loc39)
    cf.cond_br %105, ^bb59, ^bb63 loc(#loc39)
  ^bb59:  // pred: ^bb58
    cf.br ^bb60(%c0 : index) loc(#loc39)
  ^bb60(%106: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:40)):  // 2 preds: ^bb59, ^bb61
    %107 = arith.cmpi slt, %106, %c32 : index loc(#loc39)
    cf.cond_br %107, ^bb61, ^bb62 loc(#loc39)
  ^bb61:  // pred: ^bb60
    memref.store %69, %alloc_22[%104, %106] : memref<1x32xi32> loc(#loc39)
    %108 = arith.addi %106, %c1 : index loc(#loc39)
    cf.br ^bb60(%108 : index) loc(#loc39)
  ^bb62:  // pred: ^bb60
    %109 = arith.addi %104, %c1 : index loc(#loc39)
    cf.br ^bb58(%109 : index) loc(#loc39)
  ^bb63:  // pred: ^bb58
    %reinterpret_cast_28 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<64xi1> to memref<64x1xi1> loc(#loc40)
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1> loc(#loc41)
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1> loc(#loc41)
    cf.br ^bb64(%c0 : index) loc(#loc41)
  ^bb64(%110: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb63, ^bb68
    %111 = arith.cmpi slt, %110, %c64 : index loc(#loc41)
    cf.cond_br %111, ^bb65, ^bb69 loc(#loc41)
  ^bb65:  // pred: ^bb64
    cf.br ^bb66(%c0 : index) loc(#loc41)
  ^bb66(%112: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb65, ^bb67
    %113 = arith.cmpi slt, %112, %c32 : index loc(#loc41)
    cf.cond_br %113, ^bb67, ^bb68 loc(#loc41)
  ^bb67:  // pred: ^bb66
    %114 = memref.load %reinterpret_cast_28[%110, %c0] : memref<64x1xi1> loc(#loc41)
    memref.store %114, %alloc_30[%110, %112] : memref<64x32xi1> loc(#loc41)
    %115 = arith.addi %112, %c1 : index loc(#loc41)
    cf.br ^bb66(%115 : index) loc(#loc41)
  ^bb68:  // pred: ^bb66
    %116 = arith.addi %110, %c1 : index loc(#loc41)
    cf.br ^bb64(%116 : index) loc(#loc41)
  ^bb69:  // pred: ^bb64
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc4)
    cf.br ^bb70(%c0 : index) loc(#loc4)
  ^bb70(%117: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:27)):  // 2 preds: ^bb69, ^bb74
    %118 = arith.cmpi slt, %117, %c64 : index loc(#loc4)
    cf.cond_br %118, ^bb71, ^bb75 loc(#loc4)
  ^bb71:  // pred: ^bb70
    cf.br ^bb72(%c0 : index) loc(#loc4)
  ^bb72(%119: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:27)):  // 2 preds: ^bb71, ^bb73
    %120 = arith.cmpi slt, %119, %c32 : index loc(#loc4)
    cf.cond_br %120, ^bb73, ^bb74 loc(#loc4)
  ^bb73:  // pred: ^bb72
    memref.store %4, %alloc_31[%117, %119] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc4)
    %121 = arith.addi %119, %c1 : index loc(#loc4)
    cf.br ^bb72(%121 : index) loc(#loc4)
  ^bb74:  // pred: ^bb72
    %122 = arith.addi %117, %c1 : index loc(#loc4)
    cf.br ^bb70(%122 : index) loc(#loc4)
  ^bb75:  // pred: ^bb70
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32> loc(#loc42)
    memref.copy %alloc_2, %alloc_32 : memref<64x32xf32> to memref<64x32xf32> loc(#loc42)
    cf.br ^bb76(%67, %alloc_32 : i32, memref<64x32xf32>) loc(#loc42)
  ^bb76(%123: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":165:59), %124: memref<64x32xf32> loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":165:59)):  // 2 preds: ^bb75, ^bb172
    %125 = arith.cmpi slt, %123, %69 : i32 loc(#loc42)
    cf.cond_br %125, ^bb77, ^bb173 loc(#loc42)
  ^bb77:  // pred: ^bb76
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc43)
    cf.br ^bb78(%c0 : index) loc(#loc43)
  ^bb78(%126: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":167:31)):  // 2 preds: ^bb77, ^bb79
    %127 = arith.cmpi slt, %126, %c32 : index loc(#loc43)
    cf.cond_br %127, ^bb79, ^bb80 loc(#loc43)
  ^bb79:  // pred: ^bb78
    memref.store %123, %alloc_33[%126] : memref<32xi32> loc(#loc43)
    %128 = arith.addi %126, %c1 : index loc(#loc43)
    cf.br ^bb78(%128 : index) loc(#loc43)
  ^bb80:  // pred: ^bb78
    cf.br ^bb81(%c0 : index) loc(#loc43)
  ^bb81(%129: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":167:31)):  // 2 preds: ^bb80, ^bb82
    %130 = arith.cmpi slt, %129, %c32 : index loc(#loc43)
    cf.cond_br %130, ^bb82, ^bb83 loc(#loc43)
  ^bb82:  // pred: ^bb81
    %131 = memref.load %alloc_33[%129] : memref<32xi32> loc(#loc43)
    %132 = memref.load %alloc_19[%129] : memref<32xi32> loc(#loc43)
    %133 = arith.addi %131, %132 : i32 loc(#loc43)
    memref.store %133, %alloc_33[%129] : memref<32xi32> loc(#loc43)
    %134 = arith.addi %129, %c1 : index loc(#loc43)
    cf.br ^bb81(%134 : index) loc(#loc43)
  ^bb83:  // pred: ^bb81
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<32xi1> loc(#loc31)
    cf.br ^bb84(%c0 : index) loc(#loc31)
  ^bb84(%135: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":173:30)):  // 2 preds: ^bb83, ^bb85
    %136 = arith.cmpi slt, %135, %c32 : index loc(#loc31)
    cf.cond_br %136, ^bb85, ^bb86 loc(#loc31)
  ^bb85:  // pred: ^bb84
    %137 = memref.load %alloc_33[%135] : memref<32xi32> loc(#loc31)
    %138 = memref.load %alloc_20[%135] : memref<32xi32> loc(#loc31)
    %139 = arith.cmpi slt, %137, %138 : i32 loc(#loc31)
    memref.store %139, %alloc_34[%135] : memref<32xi1> loc(#loc31)
    %140 = arith.addi %135, %c1 : index loc(#loc31)
    cf.br ^bb84(%140 : index) loc(#loc31)
  ^bb86:  // pred: ^bb84
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc44)
    cf.br ^bb87(%c0 : index) loc(#loc44)
  ^bb87(%141: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":172:26)):  // 2 preds: ^bb86, ^bb88
    %142 = arith.cmpi slt, %141, %c32 : index loc(#loc44)
    cf.cond_br %142, ^bb88, ^bb89 loc(#loc44)
  ^bb88:  // pred: ^bb87
    %143 = memref.load %alloc_33[%141] : memref<32xi32> loc(#loc44)
    %144 = memref.load %alloc_5[%141] : memref<32xi32> loc(#loc44)
    %145 = arith.divsi %143, %144 : i32 loc(#loc44)
    memref.store %145, %alloc_35[%141] : memref<32xi32> loc(#loc44)
    %146 = arith.addi %141, %c1 : index loc(#loc44)
    cf.br ^bb87(%146 : index) loc(#loc44)
  ^bb89:  // pred: ^bb87
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc33)
    cf.br ^bb90(%c0 : index) loc(#loc33)
  ^bb90(%147: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":172:16)):  // 2 preds: ^bb89, ^bb91
    %148 = arith.cmpi slt, %147, %c32 : index loc(#loc33)
    cf.cond_br %148, ^bb91, ^bb92 loc(#loc33)
  ^bb91:  // pred: ^bb90
    %149 = memref.load %alloc_21[%147] : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc33)
    %150 = memref.load %alloc_35[%147] : memref<32xi32> loc(#loc33)
    %151 = arith.muli %150, %1 : i32 loc(#loc33)
    %152 = tptr.ptradd %149 %151 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc33)
    memref.store %152, %alloc_36[%147] : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc33)
    %153 = arith.addi %147, %c1 : index loc(#loc33)
    cf.br ^bb90(%153 : index) loc(#loc33)
  ^bb92:  // pred: ^bb90
    cf.br ^bb93(%c0 : index) loc(#loc45)
  ^bb93(%154: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":171:16)):  // 2 preds: ^bb92, ^bb98
    %155 = arith.cmpi slt, %154, %c32 : index loc(#loc45)
    cf.cond_br %155, ^bb94, ^bb99 loc(#loc45)
  ^bb94:  // pred: ^bb93
    %156 = memref.load %alloc_36[%154] : memref<32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc45)
    %157 = memref.load %alloc_34[%154] : memref<32xi1> loc(#loc45)
    %158 = memref.load %alloc_4[%154] : memref<32xi32> loc(#loc45)
    %159 = tptr.to_memref %156 : <#tptr.default_memory_space> to memref<1xi32> loc(#loc45)
    cf.cond_br %157, ^bb95, ^bb96 loc(#loc45)
  ^bb95:  // pred: ^bb94
    %160 = memref.load %159[%c0] : memref<1xi32> loc(#loc45)
    cf.br ^bb97(%160 : i32) loc(#loc45)
  ^bb96:  // pred: ^bb94
    cf.br ^bb97(%158 : i32) loc(#loc45)
  ^bb97(%161: i32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":171:16)):  // 2 preds: ^bb95, ^bb96
    cf.br ^bb98 loc(#loc45)
  ^bb98:  // pred: ^bb97
    memref.store %161, %alloc_3[%154] : memref<32xi32> loc(#loc45)
    %162 = arith.addi %154, %c1 : index loc(#loc45)
    cf.br ^bb93(%162 : index) loc(#loc45)
  ^bb99:  // pred: ^bb93
    cf.br ^bb100(%c0 : index) loc(#loc46)
  ^bb100(%163: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:38)):  // 2 preds: ^bb99, ^bb101
    %164 = arith.cmpi slt, %163, %c32 : index loc(#loc46)
    cf.cond_br %164, ^bb101, ^bb102 loc(#loc46)
  ^bb101:  // pred: ^bb100
    %165 = memref.load %alloc_3[%163] : memref<32xi32> loc(#loc46)
    %166 = memref.load %alloc_5[%163] : memref<32xi32> loc(#loc46)
    %167 = arith.muli %165, %166 : i32 loc(#loc46)
    memref.store %167, %alloc_3[%163] : memref<32xi32> loc(#loc46)
    %168 = arith.addi %163, %c1 : index loc(#loc46)
    cf.br ^bb100(%168 : index) loc(#loc46)
  ^bb102:  // pred: ^bb100
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<32xi32> loc(#loc47)
    cf.br ^bb103(%c0 : index) loc(#loc47)
  ^bb103(%169: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:59)):  // 2 preds: ^bb102, ^bb104
    %170 = arith.cmpi slt, %169, %c32 : index loc(#loc47)
    cf.cond_br %170, ^bb104, ^bb105 loc(#loc47)
  ^bb104:  // pred: ^bb103
    %171 = memref.load %alloc_33[%169] : memref<32xi32> loc(#loc47)
    %172 = memref.load %alloc_5[%169] : memref<32xi32> loc(#loc47)
    %173 = arith.remsi %171, %172 : i32 loc(#loc47)
    memref.store %173, %alloc_37[%169] : memref<32xi32> loc(#loc47)
    %174 = arith.addi %169, %c1 : index loc(#loc47)
    cf.br ^bb103(%174 : index) loc(#loc47)
  ^bb105:  // pred: ^bb103
    cf.br ^bb106(%c0 : index) loc(#loc48)
  ^bb106(%175: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":177:50)):  // 2 preds: ^bb105, ^bb107
    %176 = arith.cmpi slt, %175, %c32 : index loc(#loc48)
    cf.cond_br %176, ^bb107, ^bb108 loc(#loc48)
  ^bb107:  // pred: ^bb106
    %177 = memref.load %alloc_3[%175] : memref<32xi32> loc(#loc48)
    %178 = memref.load %alloc_37[%175] : memref<32xi32> loc(#loc48)
    %179 = arith.addi %177, %178 : i32 loc(#loc48)
    memref.store %179, %alloc_3[%175] : memref<32xi32> loc(#loc48)
    %180 = arith.addi %175, %c1 : index loc(#loc48)
    cf.br ^bb106(%180 : index) loc(#loc48)
  ^bb108:  // pred: ^bb106
    %reinterpret_cast_38 = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<32xi32> to memref<1x32xi32> loc(#loc49)
    cf.br ^bb109(%c0 : index) loc(#loc34)
  ^bb109(%181: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:44)):  // 2 preds: ^bb108, ^bb113
    %182 = arith.cmpi slt, %181, %c1 : index loc(#loc34)
    cf.cond_br %182, ^bb110, ^bb114 loc(#loc34)
  ^bb110:  // pred: ^bb109
    cf.br ^bb111(%c0 : index) loc(#loc34)
  ^bb111(%183: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:44)):  // 2 preds: ^bb110, ^bb112
    %184 = arith.cmpi slt, %183, %c32 : index loc(#loc34)
    cf.cond_br %184, ^bb112, ^bb113 loc(#loc34)
  ^bb112:  // pred: ^bb111
    %185 = memref.load %reinterpret_cast_38[%181, %183] : memref<1x32xi32> loc(#loc34)
    %186 = memref.load %alloc_23[%181, %183] : memref<1x32xi32> loc(#loc34)
    %187 = arith.muli %185, %186 : i32 loc(#loc34)
    memref.store %187, %reinterpret_cast_38[%181, %183] : memref<1x32xi32> loc(#loc34)
    %188 = arith.addi %183, %c1 : index loc(#loc34)
    cf.br ^bb111(%188 : index) loc(#loc34)
  ^bb113:  // pred: ^bb111
    %189 = arith.addi %181, %c1 : index loc(#loc34)
    cf.br ^bb109(%189 : index) loc(#loc34)
  ^bb114:  // pred: ^bb109
    cf.br ^bb115(%c0 : index) loc(#loc36)
  ^bb115(%190: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:26)):  // 2 preds: ^bb114, ^bb119
    %191 = arith.cmpi slt, %190, %c1 : index loc(#loc36)
    cf.cond_br %191, ^bb116, ^bb120 loc(#loc36)
  ^bb116:  // pred: ^bb115
    cf.br ^bb117(%c0 : index) loc(#loc36)
  ^bb117(%192: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:26)):  // 2 preds: ^bb116, ^bb118
    %193 = arith.cmpi slt, %192, %c32 : index loc(#loc36)
    cf.cond_br %193, ^bb118, ^bb119 loc(#loc36)
  ^bb118:  // pred: ^bb117
    %194 = memref.load %reinterpret_cast_38[%190, %192] : memref<1x32xi32> loc(#loc36)
    %195 = memref.load %alloc_24[%190, %192] : memref<1x32xi32> loc(#loc36)
    %196 = arith.addi %194, %195 : i32 loc(#loc36)
    memref.store %196, %reinterpret_cast_38[%190, %192] : memref<1x32xi32> loc(#loc36)
    %197 = arith.addi %192, %c1 : index loc(#loc36)
    cf.br ^bb117(%197 : index) loc(#loc36)
  ^bb119:  // pred: ^bb117
    %198 = arith.addi %190, %c1 : index loc(#loc36)
    cf.br ^bb115(%198 : index) loc(#loc36)
  ^bb120:  // pred: ^bb115
    cf.br ^bb121(%c0 : index) loc(#loc38)
  ^bb121(%199: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb120, ^bb125
    %200 = arith.cmpi slt, %199, %c64 : index loc(#loc38)
    cf.cond_br %200, ^bb122, ^bb126 loc(#loc38)
  ^bb122:  // pred: ^bb121
    cf.br ^bb123(%c0 : index) loc(#loc38)
  ^bb123(%201: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb122, ^bb124
    %202 = arith.cmpi slt, %201, %c32 : index loc(#loc38)
    cf.cond_br %202, ^bb124, ^bb125 loc(#loc38)
  ^bb124:  // pred: ^bb123
    %203 = memref.load %reinterpret_cast_38[%c0, %201] : memref<1x32xi32> loc(#loc38)
    memref.store %203, %alloc_26[%199, %201] : memref<64x32xi32> loc(#loc38)
    %204 = arith.addi %201, %c1 : index loc(#loc38)
    cf.br ^bb123(%204 : index) loc(#loc38)
  ^bb125:  // pred: ^bb123
    %205 = arith.addi %199, %c1 : index loc(#loc38)
    cf.br ^bb121(%205 : index) loc(#loc38)
  ^bb126:  // pred: ^bb121
    cf.br ^bb127(%c0 : index) loc(#loc38)
  ^bb127(%206: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb126, ^bb131
    %207 = arith.cmpi slt, %206, %c64 : index loc(#loc38)
    cf.cond_br %207, ^bb128, ^bb132 loc(#loc38)
  ^bb128:  // pred: ^bb127
    cf.br ^bb129(%c0 : index) loc(#loc38)
  ^bb129(%208: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:56)):  // 2 preds: ^bb128, ^bb130
    %209 = arith.cmpi slt, %208, %c32 : index loc(#loc38)
    cf.cond_br %209, ^bb130, ^bb131 loc(#loc38)
  ^bb130:  // pred: ^bb129
    %210 = memref.load %alloc_26[%206, %208] : memref<64x32xi32> loc(#loc38)
    %211 = memref.load %alloc_27[%206, %208] : memref<64x32xi32> loc(#loc38)
    %212 = arith.addi %210, %211 : i32 loc(#loc38)
    memref.store %212, %alloc_26[%206, %208] : memref<64x32xi32> loc(#loc38)
    %213 = arith.addi %208, %c1 : index loc(#loc38)
    cf.br ^bb129(%213 : index) loc(#loc38)
  ^bb131:  // pred: ^bb129
    %214 = arith.addi %206, %c1 : index loc(#loc38)
    cf.br ^bb127(%214 : index) loc(#loc38)
  ^bb132:  // pred: ^bb127
    %reinterpret_cast_39 = memref.reinterpret_cast %alloc_33 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<32xi32> to memref<1x32xi32> loc(#loc50)
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi1> loc(#loc39)
    cf.br ^bb133(%c0 : index) loc(#loc39)
  ^bb133(%215: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:40)):  // 2 preds: ^bb132, ^bb137
    %216 = arith.cmpi slt, %215, %c1 : index loc(#loc39)
    cf.cond_br %216, ^bb134, ^bb138 loc(#loc39)
  ^bb134:  // pred: ^bb133
    cf.br ^bb135(%c0 : index) loc(#loc39)
  ^bb135(%217: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:40)):  // 2 preds: ^bb134, ^bb136
    %218 = arith.cmpi slt, %217, %c32 : index loc(#loc39)
    cf.cond_br %218, ^bb136, ^bb137 loc(#loc39)
  ^bb136:  // pred: ^bb135
    %219 = memref.load %reinterpret_cast_39[%215, %217] : memref<1x32xi32> loc(#loc39)
    %220 = memref.load %alloc_22[%215, %217] : memref<1x32xi32> loc(#loc39)
    %221 = arith.cmpi slt, %219, %220 : i32 loc(#loc39)
    memref.store %221, %alloc_40[%215, %217] : memref<1x32xi1> loc(#loc39)
    %222 = arith.addi %217, %c1 : index loc(#loc39)
    cf.br ^bb135(%222 : index) loc(#loc39)
  ^bb137:  // pred: ^bb135
    %223 = arith.addi %215, %c1 : index loc(#loc39)
    cf.br ^bb133(%223 : index) loc(#loc39)
  ^bb138:  // pred: ^bb133
    cf.br ^bb139(%c0 : index) loc(#loc41)
  ^bb139(%224: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb138, ^bb143
    %225 = arith.cmpi slt, %224, %c64 : index loc(#loc41)
    cf.cond_br %225, ^bb140, ^bb144 loc(#loc41)
  ^bb140:  // pred: ^bb139
    cf.br ^bb141(%c0 : index) loc(#loc41)
  ^bb141(%226: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb140, ^bb142
    %227 = arith.cmpi slt, %226, %c32 : index loc(#loc41)
    cf.cond_br %227, ^bb142, ^bb143 loc(#loc41)
  ^bb142:  // pred: ^bb141
    %228 = memref.load %alloc_40[%c0, %226] : memref<1x32xi1> loc(#loc41)
    memref.store %228, %alloc_29[%224, %226] : memref<64x32xi1> loc(#loc41)
    %229 = arith.addi %226, %c1 : index loc(#loc41)
    cf.br ^bb141(%229 : index) loc(#loc41)
  ^bb143:  // pred: ^bb141
    %230 = arith.addi %224, %c1 : index loc(#loc41)
    cf.br ^bb139(%230 : index) loc(#loc41)
  ^bb144:  // pred: ^bb139
    cf.br ^bb145(%c0 : index) loc(#loc41)
  ^bb145(%231: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb144, ^bb149
    %232 = arith.cmpi slt, %231, %c64 : index loc(#loc41)
    cf.cond_br %232, ^bb146, ^bb150 loc(#loc41)
  ^bb146:  // pred: ^bb145
    cf.br ^bb147(%c0 : index) loc(#loc41)
  ^bb147(%233: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:57)):  // 2 preds: ^bb146, ^bb148
    %234 = arith.cmpi slt, %233, %c32 : index loc(#loc41)
    cf.cond_br %234, ^bb148, ^bb149 loc(#loc41)
  ^bb148:  // pred: ^bb147
    %235 = memref.load %alloc_29[%231, %233] : memref<64x32xi1> loc(#loc41)
    %236 = memref.load %alloc_30[%231, %233] : memref<64x32xi1> loc(#loc41)
    %237 = arith.andi %235, %236 : i1 loc(#loc41)
    memref.store %237, %alloc_29[%231, %233] : memref<64x32xi1> loc(#loc41)
    %238 = arith.addi %233, %c1 : index loc(#loc41)
    cf.br ^bb147(%238 : index) loc(#loc41)
  ^bb149:  // pred: ^bb147
    %239 = arith.addi %231, %c1 : index loc(#loc41)
    cf.br ^bb145(%239 : index) loc(#loc41)
  ^bb150:  // pred: ^bb145
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc4)
    cf.br ^bb151(%c0 : index) loc(#loc4)
  ^bb151(%240: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:27)):  // 2 preds: ^bb150, ^bb155
    %241 = arith.cmpi slt, %240, %c64 : index loc(#loc4)
    cf.cond_br %241, ^bb152, ^bb156 loc(#loc4)
  ^bb152:  // pred: ^bb151
    cf.br ^bb153(%c0 : index) loc(#loc4)
  ^bb153(%242: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:27)):  // 2 preds: ^bb152, ^bb154
    %243 = arith.cmpi slt, %242, %c32 : index loc(#loc4)
    cf.cond_br %243, ^bb154, ^bb155 loc(#loc4)
  ^bb154:  // pred: ^bb153
    %244 = memref.load %alloc_31[%240, %242] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc4)
    %245 = memref.load %alloc_26[%240, %242] : memref<64x32xi32> loc(#loc4)
    %246 = arith.muli %245, %0 : i32 loc(#loc4)
    %247 = tptr.ptradd %244 %246 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space> loc(#loc4)
    memref.store %247, %alloc_41[%240, %242] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc4)
    %248 = arith.addi %242, %c1 : index loc(#loc4)
    cf.br ^bb153(%248 : index) loc(#loc4)
  ^bb155:  // pred: ^bb153
    %249 = arith.addi %240, %c1 : index loc(#loc4)
    cf.br ^bb151(%249 : index) loc(#loc4)
  ^bb156:  // pred: ^bb151
    cf.br ^bb157(%c0 : index) loc(#loc5)
  ^bb157(%250: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:16)):  // 2 preds: ^bb156, ^bb165
    %251 = arith.cmpi slt, %250, %c64 : index loc(#loc5)
    cf.cond_br %251, ^bb158, ^bb166 loc(#loc5)
  ^bb158:  // pred: ^bb157
    cf.br ^bb159(%c0 : index) loc(#loc5)
  ^bb159(%252: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:16)):  // 2 preds: ^bb158, ^bb164
    %253 = arith.cmpi slt, %252, %c32 : index loc(#loc5)
    cf.cond_br %253, ^bb160, ^bb165 loc(#loc5)
  ^bb160:  // pred: ^bb159
    %254 = memref.load %alloc_41[%250, %252] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>> loc(#loc5)
    %255 = memref.load %alloc_29[%250, %252] : memref<64x32xi1> loc(#loc5)
    %256 = memref.load %alloc_2[%250, %252] : memref<64x32xf32> loc(#loc5)
    %257 = tptr.to_memref %254 : <#tptr.default_memory_space> to memref<1xf32> loc(#loc5)
    cf.cond_br %255, ^bb161, ^bb162 loc(#loc5)
  ^bb161:  // pred: ^bb160
    %258 = memref.load %257[%c0] : memref<1xf32> loc(#loc5)
    cf.br ^bb163(%258 : f32) loc(#loc5)
  ^bb162:  // pred: ^bb160
    cf.br ^bb163(%256 : f32) loc(#loc5)
  ^bb163(%259: f32 loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":184:16)):  // 2 preds: ^bb161, ^bb162
    cf.br ^bb164 loc(#loc5)
  ^bb164:  // pred: ^bb163
    memref.store %259, %alloc[%250, %252] : memref<64x32xf32> loc(#loc5)
    %260 = arith.addi %252, %c1 : index loc(#loc5)
    cf.br ^bb159(%260 : index) loc(#loc5)
  ^bb165:  // pred: ^bb159
    %261 = arith.addi %250, %c1 : index loc(#loc5)
    cf.br ^bb157(%261 : index) loc(#loc5)
  ^bb166:  // pred: ^bb157
    cf.br ^bb167(%c0 : index) loc(#loc51)
  ^bb167(%262: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":188:28)):  // 2 preds: ^bb166, ^bb171
    %263 = arith.cmpi slt, %262, %c64 : index loc(#loc51)
    cf.cond_br %263, ^bb168, ^bb172 loc(#loc51)
  ^bb168:  // pred: ^bb167
    cf.br ^bb169(%c0 : index) loc(#loc51)
  ^bb169(%264: index loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":188:28)):  // 2 preds: ^bb168, ^bb170
    %265 = arith.cmpi slt, %264, %c32 : index loc(#loc51)
    cf.cond_br %265, ^bb170, ^bb171 loc(#loc51)
  ^bb170:  // pred: ^bb169
    %266 = memref.load %124[%262, %264] : memref<64x32xf32> loc(#loc51)
    %267 = memref.load %alloc[%262, %264] : memref<64x32xf32> loc(#loc51)
    %268 = arith.addf %266, %267 : f32 loc(#loc51)
    memref.store %268, %124[%262, %264] : memref<64x32xf32> loc(#loc51)
    %269 = arith.addi %264, %c1 : index loc(#loc51)
    cf.br ^bb169(%269 : index) loc(#loc51)
  ^bb171:  // pred: ^bb169
    %270 = arith.addi %262, %c1 : index loc(#loc51)
    cf.br ^bb167(%270 : index) loc(#loc51)
  ^bb172:  // pred: ^bb167
    %271 = arith.addi %123, %c32_i32 : i32 loc(#loc42)
    cf.br ^bb76(%271, %124 : i32, memref<64x32xf32>) loc(#loc42)
  ^bb173:  // pred: ^bb76
    cf.br ^bb175(%124 : memref<64x32xf32>) loc(#loc30)
  ^bb174:  // pred: ^bb29
    cf.br ^bb175(%alloc_2 : memref<64x32xf32>) loc(#loc30)
  ^bb175(%272: memref<64x32xf32> loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":164:7)):  // 2 preds: ^bb173, ^bb174
    cf.br ^bb176 loc(#loc30)
  ^bb176:  // pred: ^bb175
    %273 = arith.cmpi eq, %arg24, %c0_i32 : i32 loc(#loc52)
    %274 = arith.cmpi eq, %arg25, %c1_i32 : i32 loc(#loc53)
    %275 = arith.cmpi eq, %arg26, %c1_i32 : i32 loc(#loc54)
    %276 = arith.andi %274, %275 : i1 loc(#loc55)
    %277 = arith.andi %273, %276 : i1 loc(#loc55)
    cf.cond_br %277, ^bb177, ^bb178 loc(#loc56)
  ^bb177:  // pred: ^bb176
    %reinterpret_cast_42 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<*xf32> to memref<64x32xf32, strided<[32, 1]>> loc(#loc57)
    memref.copy %272, %reinterpret_cast_42 : memref<64x32xf32> to memref<64x32xf32, strided<[32, 1]>> loc(#loc58)
    cf.br ^bb178 loc(#loc56)
  ^bb178:  // 2 preds: ^bb176, ^bb177
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":114:43)
#loc3 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":171:32)
#loc8 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":103:29)
#loc9 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":125:21)
#loc10 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":104:39)
#loc11 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":104:44)
#loc13 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":114:32)
#loc14 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":118:25)
#loc15 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":119:20)
#loc16 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":118:58)
#loc17 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":118:38)
#loc19 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":124:29)
#loc20 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":124:65)
#loc21 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":124:45)
#loc22 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":126:13)
#loc23 = loc("/data03/yanxitan/mr/triton/python/triton/language/standard.py":40:22)
#loc24 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":144:50)
#loc25 = loc("/data03/yanxitan/mr/triton/python/triton/language/standard.py":40:28)
#loc26 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":145:40)
#loc27 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":146:47)
#loc28 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":147:30)
#loc29 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":164:22)
#loc32 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":171:57)
#loc35 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:40)
#loc37 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":182:63)
#loc40 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:64)
#loc49 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":181:33)
#loc50 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":185:29)
#loc52 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":194:20)
#loc53 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":194:41)
#loc54 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":194:62)
#loc55 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":194:47)
#loc56 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":194:7)
#loc57 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":197:23)
#loc58 = loc("/data03/yanxitan/yanxi-doing/vllm_scripts/check_K.py":197:37)
#loc59 = loc(callsite(#loc23 at #loc24))
#loc60 = loc(callsite(#loc25 at #loc24))


// CHECK-NOT: tptr.
// CHECK-NOT: ptr.ptr