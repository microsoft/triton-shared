// RUN: triton-shared-opt --tptr-to-llvm %s | FileCheck %s

module {
  func.func @_fwd_grouped_kernel_stage1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: memref<*xi32> {tt.divisibility = 16 : i32}, %arg5: memref<*xi32> {tt.divisibility = 16 : i32}, %arg6: memref<*xf32> {tt.divisibility = 16 : i32}, %arg7: memref<*xf32> {tt.divisibility = 16 : i32}, %arg8: memref<*xf32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %0 = tptr.type_offset f32  : i32
    %c0 = arith.constant 0 : index
    %1 = tptr.type_offset i32  : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cast = memref.cast %arg5 : memref<*xi32> to memref<1xi32>
    %2 = tptr.from_memref %cast : memref<1xi32> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg4 : memref<*xi32> to memref<1xi32>
    %3 = tptr.from_memref %cast_0 : memref<1xi32> to <#tptr.default_memory_space>
    %cast_1 = memref.cast %arg1 : memref<*xf32> to memref<1xf32>
    %4 = tptr.from_memref %cast_1 : memref<1xf32> to <#tptr.default_memory_space>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%5: index):  // 2 preds: ^bb0, ^bb5
    %6 = arith.cmpi slt, %5, %c64 : index
    cf.cond_br %6, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%7: index):  // 2 preds: ^bb2, ^bb4
    %8 = arith.cmpi slt, %7, %c32 : index
    cf.cond_br %8, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc_2[%5, %7] : memref<64x32xf32>
    %9 = arith.addi %7, %c1 : index
    cf.br ^bb3(%9 : index)
  ^bb5:  // pred: ^bb3
    %10 = arith.addi %5, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb6:  // pred: ^bb1
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb7(%c0 : index)
  ^bb7(%11: index):  // 2 preds: ^bb6, ^bb8
    %12 = arith.cmpi slt, %11, %c32 : index
    cf.cond_br %12, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    memref.store %c0_i32, %alloc_4[%11] : memref<32xi32>
    %13 = arith.addi %11, %c1 : index
    cf.br ^bb7(%13 : index)
  ^bb9:  // pred: ^bb7
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb10(%c0 : index)
  ^bb10(%14: index):  // 2 preds: ^bb9, ^bb11
    %15 = arith.cmpi slt, %14, %c32 : index
    cf.cond_br %15, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    memref.store %c16_i32, %alloc_5[%14] : memref<32xi32>
    %16 = arith.addi %14, %c1 : index
    cf.br ^bb10(%16 : index)
  ^bb12:  // pred: ^bb10
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
    cf.br ^bb13(%c0 : index)
  ^bb13(%17: index):  // 2 preds: ^bb12, ^bb14
    %18 = arith.cmpi slt, %17, %c64 : index
    cf.cond_br %18, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    memref.store %c64_i32, %alloc_7[%17] : memref<64xi32>
    %19 = arith.addi %17, %c1 : index
    cf.br ^bb13(%19 : index)
  ^bb15:  // pred: ^bb13
    %20 = arith.muli %arg25, %c4_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.addi %arg25, %c1_i32 : i32
    %23 = arith.muli %22, %c4_i32 : i32
    cf.br ^bb16(%c0 : index)
  ^bb16(%24: index):  // 2 preds: ^bb15, ^bb17
    %25 = arith.cmpi slt, %24, %c64 : index
    cf.cond_br %25, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %26 = arith.index_cast %24 : index to i32
    memref.store %26, %alloc_6[%24] : memref<64xi32>
    %27 = arith.addi %24, %c1 : index
    cf.br ^bb16(%27 : index)
  ^bb18:  // pred: ^bb16
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<64xi1>
    cf.br ^bb19(%c0 : index)
  ^bb19(%28: index):  // 2 preds: ^bb18, ^bb20
    %29 = arith.cmpi slt, %28, %c64 : index
    cf.cond_br %29, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %30 = memref.load %alloc_6[%28] : memref<64xi32>
    %31 = memref.load %alloc_7[%28] : memref<64xi32>
    %32 = arith.cmpi slt, %30, %31 : i32
    memref.store %32, %alloc_8[%28] : memref<64xi1>
    %33 = arith.addi %28, %c1 : index
    cf.br ^bb19(%33 : index)
  ^bb21:  // pred: ^bb19
    %34 = arith.muli %arg24, %1 : i32
    %35 = tptr.ptradd %2 %34 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    %36 = tptr.to_memref %35 : <#tptr.default_memory_space> to memref<1xi32>
    %37 = memref.load %36[%c0] : memref<1xi32>
    %38 = arith.muli %arg24, %arg10 : i32
    %39 = arith.index_cast %38 : i32 to index
    %40 = arith.index_cast %arg11 : i32 to index
    %41 = arith.muli %21, %40 : index
    %42 = arith.addi %39, %41 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%42], sizes: [4, 64], strides: [%40, 1] : memref<*xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
    %43 = arith.addi %21, %c4 : index
    %44 = arith.index_cast %23 : i32 to index
    %45 = arith.minsi %43, %44 : index
    %46 = arith.maxsi %45, %21 : index
    %47 = arith.subi %46, %21 : index
    %48 = arith.minsi %43, %c8 : index
    %49 = arith.maxsi %48, %21 : index
    %50 = arith.subi %49, %21 : index
    %51 = arith.minsi %47, %50 : index
    %52 = arith.minsi %51, %c4 : index
    %alloc_9 = memref.alloc() : memref<4x64xf32>
    %53 = arith.cmpi slt, %52, %c4 : index
    cf.cond_br %53, ^bb22, ^bb29
  ^bb22:  // pred: ^bb21
    cf.br ^bb23(%c0 : index)
  ^bb23(%54: index):  // 2 preds: ^bb22, ^bb27
    %55 = arith.cmpi slt, %54, %c4 : index
    cf.cond_br %55, ^bb24, ^bb28
  ^bb24:  // pred: ^bb23
    cf.br ^bb25(%c0 : index)
  ^bb25(%56: index):  // 2 preds: ^bb24, ^bb26
    %57 = arith.cmpi slt, %56, %c64 : index
    cf.cond_br %57, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    memref.store %cst, %alloc_9[%54, %56] : memref<4x64xf32>
    %58 = arith.addi %56, %c1 : index
    cf.br ^bb25(%58 : index)
  ^bb27:  // pred: ^bb25
    %59 = arith.addi %54, %c1 : index
    cf.br ^bb23(%59 : index)
  ^bb28:  // pred: ^bb23
    cf.br ^bb29
  ^bb29:  // 2 preds: ^bb21, ^bb28
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<4x64xf32, strided<[?, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [%52, 64], strides: [%strides#0, 1] : memref<f32> to memref<?x64xf32, strided<[?, 1], offset: ?>>
    %reinterpret_cast_11 = memref.reinterpret_cast %alloc_9 to offset: [0], sizes: [%52, 64], strides: [64, 1] : memref<4x64xf32> to memref<?x64xf32, strided<[64, 1]>>
    memref.copy %reinterpret_cast_10, %reinterpret_cast_11 : memref<?x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1]>>
    %60 = arith.muli %arg24, %arg19 : i32
    %61 = arith.index_cast %60 : i32 to index
    %62 = arith.index_cast %arg20 : i32 to index
    %63 = arith.muli %21, %62 : index
    %64 = arith.addi %61, %63 : index
    %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%64], sizes: [4, 64], strides: [%62, 1] : memref<*xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
    %reinterpret_cast_13 = memref.reinterpret_cast %alloc_9 to offset: [0], sizes: [%52, 64], strides: [64, 1] : memref<4x64xf32> to memref<?x64xf32, strided<[64, 1]>>
    %base_buffer_14, %offset_15, %sizes_16:2, %strides_17:2 = memref.extract_strided_metadata %reinterpret_cast_12 : memref<4x64xf32, strided<[?, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
    %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_14 to offset: [%offset_15], sizes: [%52, 64], strides: [%strides_17#0, 1] : memref<f32> to memref<?x64xf32, strided<[?, 1], offset: ?>>
    memref.copy %reinterpret_cast_13, %reinterpret_cast_18 : memref<?x64xf32, strided<[64, 1]>> to memref<?x64xf32, strided<[?, 1], offset: ?>>
    %65 = arith.addi %37, %c1_i32 : i32
    %66 = arith.divsi %65, %c2_i32 : i32
    %67 = arith.muli %66, %arg26 : i32
    %68 = arith.addi %67, %66 : i32
    %69 = arith.minsi %68, %37 : i32
    %70 = arith.cmpi sgt, %69, %67 : i32
    cf.cond_br %70, ^bb30, ^bb174
  ^bb30:  // pred: ^bb29
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb31(%c0 : index)
  ^bb31(%71: index):  // 2 preds: ^bb30, ^bb32
    %72 = arith.cmpi slt, %71, %c32 : index
    cf.cond_br %72, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %73 = arith.index_cast %71 : index to i32
    memref.store %73, %alloc_19[%71] : memref<32xi32>
    %74 = arith.addi %71, %c1 : index
    cf.br ^bb31(%74 : index)
  ^bb33:  // pred: ^bb31
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb34(%c0 : index)
  ^bb34(%75: index):  // 2 preds: ^bb33, ^bb35
    %76 = arith.cmpi slt, %75, %c32 : index
    cf.cond_br %76, ^bb35, ^bb36
  ^bb35:  // pred: ^bb34
    memref.store %69, %alloc_20[%75] : memref<32xi32>
    %77 = arith.addi %75, %c1 : index
    cf.br ^bb34(%77 : index)
  ^bb36:  // pred: ^bb34
    %78 = arith.muli %arg9, %arg24 : i32
    %79 = arith.muli %78, %1 : i32
    %80 = tptr.ptradd %3 %79 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    cf.br ^bb37(%c0 : index)
  ^bb37(%81: index):  // 2 preds: ^bb36, ^bb38
    %82 = arith.cmpi slt, %81, %c32 : index
    cf.cond_br %82, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    memref.store %80, %alloc_21[%81] : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    %83 = arith.addi %81, %c1 : index
    cf.br ^bb37(%83 : index)
  ^bb39:  // pred: ^bb37
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32>
    cf.br ^bb40(%c0 : index)
  ^bb40(%84: index):  // 2 preds: ^bb39, ^bb44
    %85 = arith.cmpi slt, %84, %c1 : index
    cf.cond_br %85, ^bb41, ^bb45
  ^bb41:  // pred: ^bb40
    cf.br ^bb42(%c0 : index)
  ^bb42(%86: index):  // 2 preds: ^bb41, ^bb43
    %87 = arith.cmpi slt, %86, %c32 : index
    cf.cond_br %87, ^bb43, ^bb44
  ^bb43:  // pred: ^bb42
    memref.store %arg12, %alloc_23[%84, %86] : memref<1x32xi32>
    %88 = arith.addi %86, %c1 : index
    cf.br ^bb42(%88 : index)
  ^bb44:  // pred: ^bb42
    %89 = arith.addi %84, %c1 : index
    cf.br ^bb40(%89 : index)
  ^bb45:  // pred: ^bb40
    %90 = arith.muli %arg25, %arg13 : i32
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi32>
    cf.br ^bb46(%c0 : index)
  ^bb46(%91: index):  // 2 preds: ^bb45, ^bb50
    %92 = arith.cmpi slt, %91, %c1 : index
    cf.cond_br %92, ^bb47, ^bb51
  ^bb47:  // pred: ^bb46
    cf.br ^bb48(%c0 : index)
  ^bb48(%93: index):  // 2 preds: ^bb47, ^bb49
    %94 = arith.cmpi slt, %93, %c32 : index
    cf.cond_br %94, ^bb49, ^bb50
  ^bb49:  // pred: ^bb48
    memref.store %90, %alloc_24[%91, %93] : memref<1x32xi32>
    %95 = arith.addi %93, %c1 : index
    cf.br ^bb48(%95 : index)
  ^bb50:  // pred: ^bb48
    %96 = arith.addi %91, %c1 : index
    cf.br ^bb46(%96 : index)
  ^bb51:  // pred: ^bb46
    %reinterpret_cast_25 = memref.reinterpret_cast %alloc_6 to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<64xi32> to memref<64x1xi32>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi32>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi32>
    cf.br ^bb52(%c0 : index)
  ^bb52(%97: index):  // 2 preds: ^bb51, ^bb56
    %98 = arith.cmpi slt, %97, %c64 : index
    cf.cond_br %98, ^bb53, ^bb57
  ^bb53:  // pred: ^bb52
    cf.br ^bb54(%c0 : index)
  ^bb54(%99: index):  // 2 preds: ^bb53, ^bb55
    %100 = arith.cmpi slt, %99, %c32 : index
    cf.cond_br %100, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %101 = memref.load %reinterpret_cast_25[%97, %c0] : memref<64x1xi32>
    memref.store %101, %alloc_27[%97, %99] : memref<64x32xi32>
    %102 = arith.addi %99, %c1 : index
    cf.br ^bb54(%102 : index)
  ^bb56:  // pred: ^bb54
    %103 = arith.addi %97, %c1 : index
    cf.br ^bb52(%103 : index)
  ^bb57:  // pred: ^bb52
    cf.br ^bb58(%c0 : index)
  ^bb58(%104: index):  // 2 preds: ^bb57, ^bb62
    %105 = arith.cmpi slt, %104, %c1 : index
    cf.cond_br %105, ^bb59, ^bb63
  ^bb59:  // pred: ^bb58
    cf.br ^bb60(%c0 : index)
  ^bb60(%106: index):  // 2 preds: ^bb59, ^bb61
    %107 = arith.cmpi slt, %106, %c32 : index
    cf.cond_br %107, ^bb61, ^bb62
  ^bb61:  // pred: ^bb60
    memref.store %69, %alloc_22[%104, %106] : memref<1x32xi32>
    %108 = arith.addi %106, %c1 : index
    cf.br ^bb60(%108 : index)
  ^bb62:  // pred: ^bb60
    %109 = arith.addi %104, %c1 : index
    cf.br ^bb58(%109 : index)
  ^bb63:  // pred: ^bb58
    %reinterpret_cast_28 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<64xi1> to memref<64x1xi1>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1>
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1>
    cf.br ^bb64(%c0 : index)
  ^bb64(%110: index):  // 2 preds: ^bb63, ^bb68
    %111 = arith.cmpi slt, %110, %c64 : index
    cf.cond_br %111, ^bb65, ^bb69
  ^bb65:  // pred: ^bb64
    cf.br ^bb66(%c0 : index)
  ^bb66(%112: index):  // 2 preds: ^bb65, ^bb67
    %113 = arith.cmpi slt, %112, %c32 : index
    cf.cond_br %113, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %114 = memref.load %reinterpret_cast_28[%110, %c0] : memref<64x1xi1>
    memref.store %114, %alloc_30[%110, %112] : memref<64x32xi1>
    %115 = arith.addi %112, %c1 : index
    cf.br ^bb66(%115 : index)
  ^bb68:  // pred: ^bb66
    %116 = arith.addi %110, %c1 : index
    cf.br ^bb64(%116 : index)
  ^bb69:  // pred: ^bb64
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    cf.br ^bb70(%c0 : index)
  ^bb70(%117: index):  // 2 preds: ^bb69, ^bb74
    %118 = arith.cmpi slt, %117, %c64 : index
    cf.cond_br %118, ^bb71, ^bb75
  ^bb71:  // pred: ^bb70
    cf.br ^bb72(%c0 : index)
  ^bb72(%119: index):  // 2 preds: ^bb71, ^bb73
    %120 = arith.cmpi slt, %119, %c32 : index
    cf.cond_br %120, ^bb73, ^bb74
  ^bb73:  // pred: ^bb72
    memref.store %4, %alloc_31[%117, %119] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    %121 = arith.addi %119, %c1 : index
    cf.br ^bb72(%121 : index)
  ^bb74:  // pred: ^bb72
    %122 = arith.addi %117, %c1 : index
    cf.br ^bb70(%122 : index)
  ^bb75:  // pred: ^bb70
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<64x32xf32>
    memref.copy %alloc_2, %alloc_32 : memref<64x32xf32> to memref<64x32xf32>
    cf.br ^bb76(%67, %alloc_32 : i32, memref<64x32xf32>)
  ^bb76(%123: i32, %124: memref<64x32xf32>):  // 2 preds: ^bb75, ^bb172
    %125 = arith.cmpi slt, %123, %69 : i32
    cf.cond_br %125, ^bb77, ^bb173
  ^bb77:  // pred: ^bb76
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb78(%c0 : index)
  ^bb78(%126: index):  // 2 preds: ^bb77, ^bb79
    %127 = arith.cmpi slt, %126, %c32 : index
    cf.cond_br %127, ^bb79, ^bb80
  ^bb79:  // pred: ^bb78
    memref.store %123, %alloc_33[%126] : memref<32xi32>
    %128 = arith.addi %126, %c1 : index
    cf.br ^bb78(%128 : index)
  ^bb80:  // pred: ^bb78
    cf.br ^bb81(%c0 : index)
  ^bb81(%129: index):  // 2 preds: ^bb80, ^bb82
    %130 = arith.cmpi slt, %129, %c32 : index
    cf.cond_br %130, ^bb82, ^bb83
  ^bb82:  // pred: ^bb81
    %131 = memref.load %alloc_33[%129] : memref<32xi32>
    %132 = memref.load %alloc_19[%129] : memref<32xi32>
    %133 = arith.addi %131, %132 : i32
    memref.store %133, %alloc_33[%129] : memref<32xi32>
    %134 = arith.addi %129, %c1 : index
    cf.br ^bb81(%134 : index)
  ^bb83:  // pred: ^bb81
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<32xi1>
    cf.br ^bb84(%c0 : index)
  ^bb84(%135: index):  // 2 preds: ^bb83, ^bb85
    %136 = arith.cmpi slt, %135, %c32 : index
    cf.cond_br %136, ^bb85, ^bb86
  ^bb85:  // pred: ^bb84
    %137 = memref.load %alloc_33[%135] : memref<32xi32>
    %138 = memref.load %alloc_20[%135] : memref<32xi32>
    %139 = arith.cmpi slt, %137, %138 : i32
    memref.store %139, %alloc_34[%135] : memref<32xi1>
    %140 = arith.addi %135, %c1 : index
    cf.br ^bb84(%140 : index)
  ^bb86:  // pred: ^bb84
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb87(%c0 : index)
  ^bb87(%141: index):  // 2 preds: ^bb86, ^bb88
    %142 = arith.cmpi slt, %141, %c32 : index
    cf.cond_br %142, ^bb88, ^bb89
  ^bb88:  // pred: ^bb87
    %143 = memref.load %alloc_33[%141] : memref<32xi32>
    %144 = memref.load %alloc_5[%141] : memref<32xi32>
    %145 = arith.divsi %143, %144 : i32
    memref.store %145, %alloc_35[%141] : memref<32xi32>
    %146 = arith.addi %141, %c1 : index
    cf.br ^bb87(%146 : index)
  ^bb89:  // pred: ^bb87
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    cf.br ^bb90(%c0 : index)
  ^bb90(%147: index):  // 2 preds: ^bb89, ^bb91
    %148 = arith.cmpi slt, %147, %c32 : index
    cf.cond_br %148, ^bb91, ^bb92
  ^bb91:  // pred: ^bb90
    %149 = memref.load %alloc_21[%147] : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    %150 = memref.load %alloc_35[%147] : memref<32xi32>
    %151 = arith.muli %150, %1 : i32
    %152 = tptr.ptradd %149 %151 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %152, %alloc_36[%147] : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    %153 = arith.addi %147, %c1 : index
    cf.br ^bb90(%153 : index)
  ^bb92:  // pred: ^bb90
    cf.br ^bb93(%c0 : index)
  ^bb93(%154: index):  // 2 preds: ^bb92, ^bb98
    %155 = arith.cmpi slt, %154, %c32 : index
    cf.cond_br %155, ^bb94, ^bb99
  ^bb94:  // pred: ^bb93
    %156 = memref.load %alloc_36[%154] : memref<32x!ptr.ptr<#tptr.default_memory_space>>
    %157 = memref.load %alloc_34[%154] : memref<32xi1>
    %158 = memref.load %alloc_4[%154] : memref<32xi32>
    %159 = tptr.to_memref %156 : <#tptr.default_memory_space> to memref<1xi32>
    cf.cond_br %157, ^bb95, ^bb96
  ^bb95:  // pred: ^bb94
    %160 = memref.load %159[%c0] : memref<1xi32>
    cf.br ^bb97(%160 : i32)
  ^bb96:  // pred: ^bb94
    cf.br ^bb97(%158 : i32)
  ^bb97(%161: i32):  // 2 preds: ^bb95, ^bb96
    cf.br ^bb98
  ^bb98:  // pred: ^bb97
    memref.store %161, %alloc_3[%154] : memref<32xi32>
    %162 = arith.addi %154, %c1 : index
    cf.br ^bb93(%162 : index)
  ^bb99:  // pred: ^bb93
    cf.br ^bb100(%c0 : index)
  ^bb100(%163: index):  // 2 preds: ^bb99, ^bb101
    %164 = arith.cmpi slt, %163, %c32 : index
    cf.cond_br %164, ^bb101, ^bb102
  ^bb101:  // pred: ^bb100
    %165 = memref.load %alloc_3[%163] : memref<32xi32>
    %166 = memref.load %alloc_5[%163] : memref<32xi32>
    %167 = arith.muli %165, %166 : i32
    memref.store %167, %alloc_3[%163] : memref<32xi32>
    %168 = arith.addi %163, %c1 : index
    cf.br ^bb100(%168 : index)
  ^bb102:  // pred: ^bb100
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    cf.br ^bb103(%c0 : index)
  ^bb103(%169: index):  // 2 preds: ^bb102, ^bb104
    %170 = arith.cmpi slt, %169, %c32 : index
    cf.cond_br %170, ^bb104, ^bb105
  ^bb104:  // pred: ^bb103
    %171 = memref.load %alloc_33[%169] : memref<32xi32>
    %172 = memref.load %alloc_5[%169] : memref<32xi32>
    %173 = arith.remsi %171, %172 : i32
    memref.store %173, %alloc_37[%169] : memref<32xi32>
    %174 = arith.addi %169, %c1 : index
    cf.br ^bb103(%174 : index)
  ^bb105:  // pred: ^bb103
    cf.br ^bb106(%c0 : index)
  ^bb106(%175: index):  // 2 preds: ^bb105, ^bb107
    %176 = arith.cmpi slt, %175, %c32 : index
    cf.cond_br %176, ^bb107, ^bb108
  ^bb107:  // pred: ^bb106
    %177 = memref.load %alloc_3[%175] : memref<32xi32>
    %178 = memref.load %alloc_37[%175] : memref<32xi32>
    %179 = arith.addi %177, %178 : i32
    memref.store %179, %alloc_3[%175] : memref<32xi32>
    %180 = arith.addi %175, %c1 : index
    cf.br ^bb106(%180 : index)
  ^bb108:  // pred: ^bb106
    %reinterpret_cast_38 = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<32xi32> to memref<1x32xi32>
    cf.br ^bb109(%c0 : index)
  ^bb109(%181: index):  // 2 preds: ^bb108, ^bb113
    %182 = arith.cmpi slt, %181, %c1 : index
    cf.cond_br %182, ^bb110, ^bb114
  ^bb110:  // pred: ^bb109
    cf.br ^bb111(%c0 : index)
  ^bb111(%183: index):  // 2 preds: ^bb110, ^bb112
    %184 = arith.cmpi slt, %183, %c32 : index
    cf.cond_br %184, ^bb112, ^bb113
  ^bb112:  // pred: ^bb111
    %185 = memref.load %reinterpret_cast_38[%181, %183] : memref<1x32xi32>
    %186 = memref.load %alloc_23[%181, %183] : memref<1x32xi32>
    %187 = arith.muli %185, %186 : i32
    memref.store %187, %reinterpret_cast_38[%181, %183] : memref<1x32xi32>
    %188 = arith.addi %183, %c1 : index
    cf.br ^bb111(%188 : index)
  ^bb113:  // pred: ^bb111
    %189 = arith.addi %181, %c1 : index
    cf.br ^bb109(%189 : index)
  ^bb114:  // pred: ^bb109
    cf.br ^bb115(%c0 : index)
  ^bb115(%190: index):  // 2 preds: ^bb114, ^bb119
    %191 = arith.cmpi slt, %190, %c1 : index
    cf.cond_br %191, ^bb116, ^bb120
  ^bb116:  // pred: ^bb115
    cf.br ^bb117(%c0 : index)
  ^bb117(%192: index):  // 2 preds: ^bb116, ^bb118
    %193 = arith.cmpi slt, %192, %c32 : index
    cf.cond_br %193, ^bb118, ^bb119
  ^bb118:  // pred: ^bb117
    %194 = memref.load %reinterpret_cast_38[%190, %192] : memref<1x32xi32>
    %195 = memref.load %alloc_24[%190, %192] : memref<1x32xi32>
    %196 = arith.addi %194, %195 : i32
    memref.store %196, %reinterpret_cast_38[%190, %192] : memref<1x32xi32>
    %197 = arith.addi %192, %c1 : index
    cf.br ^bb117(%197 : index)
  ^bb119:  // pred: ^bb117
    %198 = arith.addi %190, %c1 : index
    cf.br ^bb115(%198 : index)
  ^bb120:  // pred: ^bb115
    cf.br ^bb121(%c0 : index)
  ^bb121(%199: index):  // 2 preds: ^bb120, ^bb125
    %200 = arith.cmpi slt, %199, %c64 : index
    cf.cond_br %200, ^bb122, ^bb126
  ^bb122:  // pred: ^bb121
    cf.br ^bb123(%c0 : index)
  ^bb123(%201: index):  // 2 preds: ^bb122, ^bb124
    %202 = arith.cmpi slt, %201, %c32 : index
    cf.cond_br %202, ^bb124, ^bb125
  ^bb124:  // pred: ^bb123
    %203 = memref.load %reinterpret_cast_38[%c0, %201] : memref<1x32xi32>
    memref.store %203, %alloc_26[%199, %201] : memref<64x32xi32>
    %204 = arith.addi %201, %c1 : index
    cf.br ^bb123(%204 : index)
  ^bb125:  // pred: ^bb123
    %205 = arith.addi %199, %c1 : index
    cf.br ^bb121(%205 : index)
  ^bb126:  // pred: ^bb121
    cf.br ^bb127(%c0 : index)
  ^bb127(%206: index):  // 2 preds: ^bb126, ^bb131
    %207 = arith.cmpi slt, %206, %c64 : index
    cf.cond_br %207, ^bb128, ^bb132
  ^bb128:  // pred: ^bb127
    cf.br ^bb129(%c0 : index)
  ^bb129(%208: index):  // 2 preds: ^bb128, ^bb130
    %209 = arith.cmpi slt, %208, %c32 : index
    cf.cond_br %209, ^bb130, ^bb131
  ^bb130:  // pred: ^bb129
    %210 = memref.load %alloc_26[%206, %208] : memref<64x32xi32>
    %211 = memref.load %alloc_27[%206, %208] : memref<64x32xi32>
    %212 = arith.addi %210, %211 : i32
    memref.store %212, %alloc_26[%206, %208] : memref<64x32xi32>
    %213 = arith.addi %208, %c1 : index
    cf.br ^bb129(%213 : index)
  ^bb131:  // pred: ^bb129
    %214 = arith.addi %206, %c1 : index
    cf.br ^bb127(%214 : index)
  ^bb132:  // pred: ^bb127
    %reinterpret_cast_39 = memref.reinterpret_cast %alloc_33 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<32xi32> to memref<1x32xi32>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi1>
    cf.br ^bb133(%c0 : index)
  ^bb133(%215: index):  // 2 preds: ^bb132, ^bb137
    %216 = arith.cmpi slt, %215, %c1 : index
    cf.cond_br %216, ^bb134, ^bb138
  ^bb134:  // pred: ^bb133
    cf.br ^bb135(%c0 : index)
  ^bb135(%217: index):  // 2 preds: ^bb134, ^bb136
    %218 = arith.cmpi slt, %217, %c32 : index
    cf.cond_br %218, ^bb136, ^bb137
  ^bb136:  // pred: ^bb135
    %219 = memref.load %reinterpret_cast_39[%215, %217] : memref<1x32xi32>
    %220 = memref.load %alloc_22[%215, %217] : memref<1x32xi32>
    %221 = arith.cmpi slt, %219, %220 : i32
    memref.store %221, %alloc_40[%215, %217] : memref<1x32xi1>
    %222 = arith.addi %217, %c1 : index
    cf.br ^bb135(%222 : index)
  ^bb137:  // pred: ^bb135
    %223 = arith.addi %215, %c1 : index
    cf.br ^bb133(%223 : index)
  ^bb138:  // pred: ^bb133
    cf.br ^bb139(%c0 : index)
  ^bb139(%224: index):  // 2 preds: ^bb138, ^bb143
    %225 = arith.cmpi slt, %224, %c64 : index
    cf.cond_br %225, ^bb140, ^bb144
  ^bb140:  // pred: ^bb139
    cf.br ^bb141(%c0 : index)
  ^bb141(%226: index):  // 2 preds: ^bb140, ^bb142
    %227 = arith.cmpi slt, %226, %c32 : index
    cf.cond_br %227, ^bb142, ^bb143
  ^bb142:  // pred: ^bb141
    %228 = memref.load %alloc_40[%c0, %226] : memref<1x32xi1>
    memref.store %228, %alloc_29[%224, %226] : memref<64x32xi1>
    %229 = arith.addi %226, %c1 : index
    cf.br ^bb141(%229 : index)
  ^bb143:  // pred: ^bb141
    %230 = arith.addi %224, %c1 : index
    cf.br ^bb139(%230 : index)
  ^bb144:  // pred: ^bb139
    cf.br ^bb145(%c0 : index)
  ^bb145(%231: index):  // 2 preds: ^bb144, ^bb149
    %232 = arith.cmpi slt, %231, %c64 : index
    cf.cond_br %232, ^bb146, ^bb150
  ^bb146:  // pred: ^bb145
    cf.br ^bb147(%c0 : index)
  ^bb147(%233: index):  // 2 preds: ^bb146, ^bb148
    %234 = arith.cmpi slt, %233, %c32 : index
    cf.cond_br %234, ^bb148, ^bb149
  ^bb148:  // pred: ^bb147
    %235 = memref.load %alloc_29[%231, %233] : memref<64x32xi1>
    %236 = memref.load %alloc_30[%231, %233] : memref<64x32xi1>
    %237 = arith.andi %235, %236 : i1
    memref.store %237, %alloc_29[%231, %233] : memref<64x32xi1>
    %238 = arith.addi %233, %c1 : index
    cf.br ^bb147(%238 : index)
  ^bb149:  // pred: ^bb147
    %239 = arith.addi %231, %c1 : index
    cf.br ^bb145(%239 : index)
  ^bb150:  // pred: ^bb145
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    cf.br ^bb151(%c0 : index)
  ^bb151(%240: index):  // 2 preds: ^bb150, ^bb155
    %241 = arith.cmpi slt, %240, %c64 : index
    cf.cond_br %241, ^bb152, ^bb156
  ^bb152:  // pred: ^bb151
    cf.br ^bb153(%c0 : index)
  ^bb153(%242: index):  // 2 preds: ^bb152, ^bb154
    %243 = arith.cmpi slt, %242, %c32 : index
    cf.cond_br %243, ^bb154, ^bb155
  ^bb154:  // pred: ^bb153
    %244 = memref.load %alloc_31[%240, %242] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    %245 = memref.load %alloc_26[%240, %242] : memref<64x32xi32>
    %246 = arith.muli %245, %0 : i32
    %247 = tptr.ptradd %244 %246 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
    memref.store %247, %alloc_41[%240, %242] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    %248 = arith.addi %242, %c1 : index
    cf.br ^bb153(%248 : index)
  ^bb155:  // pred: ^bb153
    %249 = arith.addi %240, %c1 : index
    cf.br ^bb151(%249 : index)
  ^bb156:  // pred: ^bb151
    cf.br ^bb157(%c0 : index)
  ^bb157(%250: index):  // 2 preds: ^bb156, ^bb165
    %251 = arith.cmpi slt, %250, %c64 : index
    cf.cond_br %251, ^bb158, ^bb166
  ^bb158:  // pred: ^bb157
    cf.br ^bb159(%c0 : index)
  ^bb159(%252: index):  // 2 preds: ^bb158, ^bb164
    %253 = arith.cmpi slt, %252, %c32 : index
    cf.cond_br %253, ^bb160, ^bb165
  ^bb160:  // pred: ^bb159
    %254 = memref.load %alloc_41[%250, %252] : memref<64x32x!ptr.ptr<#tptr.default_memory_space>>
    %255 = memref.load %alloc_29[%250, %252] : memref<64x32xi1>
    %256 = memref.load %alloc_2[%250, %252] : memref<64x32xf32>
    %257 = tptr.to_memref %254 : <#tptr.default_memory_space> to memref<1xf32>
    cf.cond_br %255, ^bb161, ^bb162
  ^bb161:  // pred: ^bb160
    %258 = memref.load %257[%c0] : memref<1xf32>
    cf.br ^bb163(%258 : f32)
  ^bb162:  // pred: ^bb160
    cf.br ^bb163(%256 : f32)
  ^bb163(%259: f32):  // 2 preds: ^bb161, ^bb162
    cf.br ^bb164
  ^bb164:  // pred: ^bb163
    memref.store %259, %alloc[%250, %252] : memref<64x32xf32>
    %260 = arith.addi %252, %c1 : index
    cf.br ^bb159(%260 : index)
  ^bb165:  // pred: ^bb159
    %261 = arith.addi %250, %c1 : index
    cf.br ^bb157(%261 : index)
  ^bb166:  // pred: ^bb157
    cf.br ^bb167(%c0 : index)
  ^bb167(%262: index):  // 2 preds: ^bb166, ^bb171
    %263 = arith.cmpi slt, %262, %c64 : index
    cf.cond_br %263, ^bb168, ^bb172
  ^bb168:  // pred: ^bb167
    cf.br ^bb169(%c0 : index)
  ^bb169(%264: index):  // 2 preds: ^bb168, ^bb170
    %265 = arith.cmpi slt, %264, %c32 : index
    cf.cond_br %265, ^bb170, ^bb171
  ^bb170:  // pred: ^bb169
    %266 = memref.load %124[%262, %264] : memref<64x32xf32>
    %267 = memref.load %alloc[%262, %264] : memref<64x32xf32>
    %268 = arith.addf %266, %267 : f32
    memref.store %268, %124[%262, %264] : memref<64x32xf32>
    %269 = arith.addi %264, %c1 : index
    cf.br ^bb169(%269 : index)
  ^bb171:  // pred: ^bb169
    %270 = arith.addi %262, %c1 : index
    cf.br ^bb167(%270 : index)
  ^bb172:  // pred: ^bb167
    %271 = arith.addi %123, %c32_i32 : i32
    cf.br ^bb76(%271, %124 : i32, memref<64x32xf32>)
  ^bb173:  // pred: ^bb76
    cf.br ^bb175(%124 : memref<64x32xf32>)
  ^bb174:  // pred: ^bb29
    cf.br ^bb175(%alloc_2 : memref<64x32xf32>)
  ^bb175(%272: memref<64x32xf32>):  // 2 preds: ^bb173, ^bb174
    cf.br ^bb176
  ^bb176:  // pred: ^bb175
    %273 = arith.cmpi eq, %arg24, %c0_i32 : i32
    %274 = arith.cmpi eq, %arg25, %c1_i32 : i32
    %275 = arith.cmpi eq, %arg26, %c1_i32 : i32
    %276 = arith.andi %274, %275 : i1
    %277 = arith.andi %273, %276 : i1
    cf.cond_br %277, ^bb177, ^bb178
  ^bb177:  // pred: ^bb176
    %reinterpret_cast_42 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<*xf32> to memref<64x32xf32, strided<[32, 1]>>
    memref.copy %272, %reinterpret_cast_42 : memref<64x32xf32> to memref<64x32xf32, strided<[32, 1]>>
    cf.br ^bb178
  ^bb178:  // 2 preds: ^bb176, ^bb177
    return
  }
}

// CHECK-NOT: tptr.
// CHECK-NOT: ptr.ptr