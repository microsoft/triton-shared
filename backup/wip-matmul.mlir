module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<*xbf16> to !tt.ptr<bf16, 1>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xbf16> to !tt.ptr<bf16, 1>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<*xbf16> to !tt.ptr<bf16, 1>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.addi %arg3, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg4, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.addi %arg5, %c63_i32 : i32
    %9 = arith.divsi %8, %c64_i32 : i32
    %10 = arith.muli %7, %c8_i32 : i32
    %11 = arith.divsi %3, %10 : i32
    %12 = arith.muli %11, %c8_i32 : i32
    %13 = arith.subi %5, %12 : i32
    %14 = arith.cmpi slt, %13, %c8_i32 : i32
    %15 = arith.select %14, %13, %c8_i32 : i32
    %16 = arith.remsi %3, %15 : i32
    %17 = arith.addi %12, %16 : i32
    %18 = arith.remsi %3, %10 : i32
    %19 = arith.divsi %18, %15 : i32
    %20 = arith.muli %17, %c128_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.index_cast %20 : i32 to index
    %23 = arith.muli %19, %c256_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.index_cast %23 : i32 to index
    %26 = arith.index_cast %arg6 : i32 to index
    %27 = arith.muli %22, %26 : index
    %28 = arith.index_cast %arg7 : i32 to index
    %29 = arith.index_cast %arg8 : i32 to index
    %30 = arith.index_cast %arg9 : i32 to index
    %31 = arith.muli %25, %30 : index
    %32 = arith.muli %arg7, %c64_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.muli %arg8, %c64_i32 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36:3 = scf.for %arg12 = %c0_i32 to %9 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %27, %arg15 = %c0) -> (tensor<128x256xf32>, index, index)  : i32 {
      %55 = tts.make_tptr %1 to sizes: [64, 256], strides: [%29, %30], offsets: [%arg15, %31], shape: [0, 0], order: [] : <bf16, 1> to tensor<64x256x!tt.ptr<bf16, 1>>
      %56 = tts.make_tptr %2 to sizes: [128, 64], strides: [%26, %28], offsets: [%arg14, %c0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x64x!tt.ptr<bf16, 1>>
      %57 = "tts.load"(%56) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16, 1>>) -> tensor<128x64xbf16>
      %58 = "tts.load"(%55) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<64x256x!tt.ptr<bf16, 1>>) -> tensor<64x256xbf16>
      %59 = tt.dot %57, %58, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %60 = arith.addf %arg13, %59 : tensor<128x256xf32>
      %61 = arith.addi %arg14, %33 : index
      %62 = arith.addi %arg15, %35 : index
      scf.yield %60, %61, %62 : tensor<128x256xf32>, index, index
    }
    %37 = arith.truncf %36#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %38 = arith.index_cast %arg10 : i32 to index
    %39 = arith.muli %21, %38 : index
    %40 = arith.index_cast %arg11 : i32 to index
    %41 = arith.muli %24, %40 : index
    %42 = tts.make_tptr %0 to sizes: [128, 256], strides: [%38, %40], offsets: [%39, %41], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %43 = arith.index_cast %20 : i32 to index
    %44 = arith.addi %43, %c128 : index
    %45 = arith.index_cast %arg3 : i32 to index
    %46 = arith.minsi %44, %45 : index
    %47 = arith.subi %46, %43 : index
    %48 = arith.index_cast %23 : i32 to index
    %49 = arith.addi %48, %c256 : index
    %50 = arith.index_cast %arg4 : i32 to index
    %51 = arith.minsi %49, %50 : index
    %52 = arith.subi %51, %48 : index
    %53 = arith.minsi %47, %c128 : index
    %54 = arith.minsi %52, %c256 : index
    "tts.store"(%42, %37, %53, %54) <{static_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>, index, index) -> ()
    tt.return
  }
}

