module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: !tt.ptr<bf16, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
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
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %4, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %2, %9 : i32
    %11 = arith.cmpi slt, %10, %c8_i32 : i32
    %12 = arith.select %11, %10, %c8_i32 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %17 : i32 to index
    %20 = arith.muli %16, %c256_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.index_cast %20 : i32 to index
    %23 = arith.index_cast %arg6 : i32 to index
    %24 = arith.muli %19, %23 : index
    %25 = arith.index_cast %arg7 : i32 to index
    %26 = arith.index_cast %arg8 : i32 to index
    %27 = arith.index_cast %arg9 : i32 to index
    %28 = arith.muli %22, %27 : index
    %29 = arith.muli %arg7, %c64_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %arg8, %c64_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33:3 = scf.for %arg12 = %c0_i32 to %6 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %24, %arg15 = %c0) -> (tensor<128x256xf32>, index, index)  : i32 {
      %52 = tts.make_tptr %arg1 to sizes: [64, 256], strides: [%26, %27], offsets: [%arg15, %28], shape: [0, 0], order: [] : <bf16, 1> to tensor<64x256x!tt.ptr<bf16, 1>>
      %53 = tts.make_tptr %arg0 to sizes: [128, 64], strides: [%23, %25], offsets: [%arg14, %c0], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x64x!tt.ptr<bf16, 1>>
      %54 = "tts.load"(%53) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16, 1>>) -> tensor<128x64xbf16>
      %55 = "tts.load"(%52) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<64x256x!tt.ptr<bf16, 1>>) -> tensor<64x256xbf16>
      %56 = tt.dot %54, %55, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %57 = arith.addf %arg13, %56 : tensor<128x256xf32>
      %58 = arith.addi %arg14, %30 : index
      %59 = arith.addi %arg15, %32 : index
      scf.yield %57, %58, %59 : tensor<128x256xf32>, index, index
    }
    %34 = arith.truncf %33#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %35 = arith.index_cast %arg10 : i32 to index
    %36 = arith.muli %18, %35 : index
    %37 = arith.index_cast %arg11 : i32 to index
    %38 = arith.muli %21, %37 : index
    %39 = tts.make_tptr %arg2 to sizes: [128, 256], strides: [%35, %37], offsets: [%36, %38], shape: [0, 0], order: [] : <bf16, 1> to tensor<128x256x!tt.ptr<bf16, 1>>
    %40 = arith.index_cast %17 : i32 to index
    %41 = arith.addi %40, %c128 : index
    %42 = arith.index_cast %arg3 : i32 to index
    %43 = arith.minsi %41, %42 : index
    %44 = arith.subi %43, %40 : index
    %45 = arith.index_cast %20 : i32 to index
    %46 = arith.addi %45, %c256 : index
    %47 = arith.index_cast %arg4 : i32 to index
    %48 = arith.minsi %46, %47 : index
    %49 = arith.subi %48, %45 : index
    %50 = arith.minsi %44, %c128 : index
    %51 = arith.minsi %49, %c256 : index
    "tts.store"(%39, %34, %50, %51) <{static_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16, 1>>, tensor<128x256xbf16>, index, index) -> ()
    tt.return
  }
}

