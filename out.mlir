module {
  tt.func public @_attention_forward(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<f32>, %arg7: !tt.ptr<f32>, %arg8: !tt.ptr<i32>, %arg9: !tt.ptr<i32>, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: !tt.ptr<i32>, %arg13: !tt.ptr<i32>, %arg14: !tt.ptr<i32>, %arg15: !tt.ptr<i32>, %arg16: !tt.ptr<i32>, %arg17: !tt.ptr<i32>, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: !tt.ptr<f32>, %arg31: i32, %arg32: i32, %arg33: i32) attributes {noinline = false} {
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x16xf32>
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<1> : tensor<1x64xi32>
    %cst_2 = arith.constant dense<1> : tensor<128x64xi32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<0> : tensor<128x64xi32>
    %cst_5 = arith.constant dense<-1.000000e+06> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x16xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst_8 = arith.constant 1.44269502 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_9 = arith.constant dense<2> : tensor<128xi32>
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c128 = arith.constant 128 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %2 : i32 to i64
    %4 = "tts.create_ptr"(%arg8, %3) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %5 = tt.load %4 : !tt.ptr<i32>
    %6 = arith.shrsi %5, %c32_i32 : i32
    %7 = arith.cmpi eq, %5, %c-1_i32 : i32
    cf.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %8 = arith.extsi %5 : i32 to i64
    %9 = "tts.create_ptr"(%arg11, %8) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %10 = tt.load %9 : !tt.ptr<i32>
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.extsi %arg20 : i32 to i64
    %13 = arith.muli %11, %12 : i64
    %14 = arith.extsi %arg18 : i32 to i64
    %15 = arith.muli %1, %14 : i64
    %16 = arith.addi %13, %15 : i64
    %17 = arith.index_cast %16 : i64 to index
    %18 = arith.extsi %arg28 : i32 to i64
    %19 = arith.muli %1, %18 : i64
    %20 = arith.addi %11, %19 : i64
    %21 = arith.index_cast %20 : i64 to index
    %22 = arith.extsi %arg22 : i32 to i64
    %23 = arith.muli %1, %22 : i64
    %24 = arith.muli %6, %c128_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %27 = tt.splat %24 : i32 -> tensor<128xi32>
    %28 = arith.addi %27, %26 : tensor<128xi32>
    %29 = arith.extsi %5 : i32 to i64
    %30 = "tts.create_ptr"(%arg9, %29) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %31 = tt.load %30 : !tt.ptr<i32>
    %32 = arith.muli %31, %c2_i32 : i32
    %33 = tt.splat %32 : i32 -> tensor<128xi32>
    %34 = arith.cmpi slt, %28, %33 : tensor<128xi32>
    %35 = arith.select %34, %28, %27 : tensor<128xi1>, tensor<128xi32>
    %36 = arith.divsi %35, %cst_9 : tensor<128xi32>
    %37 = arith.remsi %35, %cst_9 : tensor<128xi32>
    %38 = arith.extsi %5 : i32 to i64
    %39 = "tts.create_ptr"(%arg10, %38) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %40 = tt.load %39 : !tt.ptr<i32>
    %41 = arith.extsi %5 : i32 to i64
    %42 = "tts.create_ptr"(%arg13, %41) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %43 = tt.load %42 : !tt.ptr<i32>
    %44 = arith.extsi %43 : i32 to i64
    %45 = arith.extsi %arg23 : i32 to i64
    %46 = arith.muli %44, %45 : i64
    %47 = arith.addi %23, %46 : i64
    %48 = arith.extsi %5 : i32 to i64
    %49 = "tts.create_ptr"(%arg15, %48) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %50 = tt.load %49 : !tt.ptr<i32>
    %51 = arith.index_cast %50 : i32 to index
    %52 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %53 = tts.make_tptr %arg16 to sizes: [64], strides: [1], offsets: [%51], shape: [0], order: [] : <i32> to tensor<64x!tt.ptr<i32>>
    %54 = "tts.load"(%53) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
    %55 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %56 = arith.divsi %11, %c2_i64 : i64
    %57 = arith.extsi %arg33 : i32 to i64
    %58 = arith.muli %56, %57 : i64
    %59 = arith.extsi %arg31 : i32 to i64
    %60 = arith.muli %1, %59 : i64
    %61 = arith.addi %58, %60 : i64
    %62 = tt.expand_dims %35 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %63 = tt.splat %arg20 : i32 -> tensor<128x1xi32>
    %64 = arith.muli %62, %63 : tensor<128x1xi32>
    %65 = tt.splat %16 : i64 -> tensor<128x1xi64>
    %66 = arith.extsi %64 : tensor<128x1xi32> to tensor<128x1xi64>
    %67 = arith.addi %65, %66 : tensor<128x1xi64>
    %68 = tt.expand_dims %55 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %69 = tt.broadcast %67 : tensor<128x1xi64> -> tensor<128x16xi64>
    %70 = tt.broadcast %68 : tensor<1x16xi32> -> tensor<128x16xi32>
    %71 = arith.extsi %70 : tensor<128x16xi32> to tensor<128x16xi64>
    %72 = arith.addi %69, %71 : tensor<128x16xi64>
    %73 = "tts.create_ptr"(%arg2, %72) : (!tt.ptr<f32>, tensor<128x16xi64>) -> tensor<128x16x!tt.ptr<f32>>
    %74 = tt.splat %16 : i64 -> tensor<128x1xi64>
    %75 = arith.extsi %64 : tensor<128x1xi32> to tensor<128x1xi64>
    %76 = arith.addi %74, %75 : tensor<128x1xi64>
    %77 = tt.broadcast %76 : tensor<128x1xi64> -> tensor<128x16xi64>
    %78 = arith.extsi %70 : tensor<128x16xi32> to tensor<128x16xi64>
    %79 = arith.addi %77, %78 : tensor<128x16xi64>
    %80 = "tts.create_ptr"(%arg0, %79) : (!tt.ptr<f32>, tensor<128x16xi64>) -> tensor<128x16x!tt.ptr<f32>>
    %81 = tt.expand_dims %54 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %82 = tt.splat %arg23 : i32 -> tensor<1x64xi32>
    %83 = arith.muli %81, %82 : tensor<1x64xi32>
    %84 = tt.splat %47 : i64 -> tensor<1x64xi64>
    %85 = arith.extsi %83 : tensor<1x64xi32> to tensor<1x64xi64>
    %86 = arith.addi %84, %85 : tensor<1x64xi64>
    %87 = tt.expand_dims %55 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %88 = tt.broadcast %86 : tensor<1x64xi64> -> tensor<16x64xi64>
    %89 = tt.broadcast %87 : tensor<16x1xi32> -> tensor<16x64xi32>
    %90 = arith.extsi %89 : tensor<16x64xi32> to tensor<16x64xi64>
    %91 = arith.addi %88, %90 : tensor<16x64xi64>
    %92 = tt.expand_dims %54 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %93 = tt.splat %arg23 : i32 -> tensor<64x1xi32>
    %94 = arith.muli %92, %93 : tensor<64x1xi32>
    %95 = tt.splat %47 : i64 -> tensor<64x1xi64>
    %96 = arith.extsi %94 : tensor<64x1xi32> to tensor<64x1xi64>
    %97 = arith.addi %95, %96 : tensor<64x1xi64>
    %98 = tt.broadcast %97 : tensor<64x1xi64> -> tensor<64x16xi64>
    %99 = tt.broadcast %68 : tensor<1x16xi32> -> tensor<64x16xi32>
    %100 = arith.extsi %99 : tensor<64x16xi32> to tensor<64x16xi64>
    %101 = arith.addi %98, %100 : tensor<64x16xi64>
    %102 = arith.mulf %arg5, %cst_8 : f32
    %103 = tt.load %73 : tensor<128x16x!tt.ptr<f32>>
    %104 = tt.splat %56 : i64 -> tensor<128xi64>
    %105 = arith.extsi %36 : tensor<128xi32> to tensor<128xi64>
    %106 = arith.addi %104, %105 : tensor<128xi64>
    %107 = "tts.create_ptr"(%arg14, %106) : (!tt.ptr<i32>, tensor<128xi64>) -> tensor<128x!tt.ptr<i32>>
    %108 = tt.load %107 : tensor<128x!tt.ptr<i32>>
    %109 = arith.muli %2, %c6_i32 : i32
    %110 = arith.extsi %109 : i32 to i64
    %111 = "tts.create_ptr"(%arg17, %110) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %112 = arith.addi %110, %c1_i64 : i64
    %113 = "tts.create_ptr"(%arg17, %112) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %114 = tt.load %111 : !tt.ptr<i32>
    %115 = tt.load %113 : !tt.ptr<i32>
    %116 = arith.muli %114, %arg23 : i32
    %117 = tt.splat %116 : i32 -> tensor<16x64xi32>
    %118 = arith.extsi %117 : tensor<16x64xi32> to tensor<16x64xi64>
    %119 = arith.addi %91, %118 : tensor<16x64xi64>
    %120 = tt.splat %116 : i32 -> tensor<64x16xi32>
    %121 = arith.extsi %120 : tensor<64x16xi32> to tensor<64x16xi64>
    %122 = arith.addi %101, %121 : tensor<64x16xi64>
    %123 = arith.subi %115, %114 : i32
    %124 = tt.expand_dims %108 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %125 = tt.broadcast %124 : tensor<128x1xi32> -> tensor<128x64xi32>
    %126 = tt.expand_dims %37 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %127 = tt.splat %arg32 : i32 -> tensor<128x1xi32>
    %128 = arith.muli %126, %127 : tensor<128x1xi32>
    %129 = tt.broadcast %128 : tensor<128x1xi32> -> tensor<128x64xi32>
    %130 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %131 = tt.splat %arg33 : i32 -> tensor<128x1xi32>
    %132 = arith.muli %130, %131 : tensor<128x1xi32>
    %133 = tt.broadcast %132 : tensor<128x1xi32> -> tensor<128x64xi32>
    %134 = tt.splat %61 : i64 -> tensor<128x64xi64>
    %135 = tt.splat %102 : f32 -> tensor<128x64xf32>
    %136 = arith.muli %arg23, %c64_i32 : i32
    %137 = tt.splat %136 : i32 -> tensor<64x16xi32>
    %138 = tt.splat %136 : i32 -> tensor<16x64xi32>
    %139:5 = scf.for %arg34 = %c0_i32 to %123 step %c64_i32 iter_args(%arg35 = %cst_7, %arg36 = %cst_6, %arg37 = %cst, %arg38 = %122, %arg39 = %119) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %211 = "tts.create_ptr"(%arg3, %arg39) : (!tt.ptr<f32>, tensor<16x64xi64>) -> tensor<16x64x!tt.ptr<f32>>
      %212 = "tts.create_ptr"(%arg4, %arg38) : (!tt.ptr<f32>, tensor<64x16xi64>) -> tensor<64x16x!tt.ptr<f32>>
      %213 = arith.addi %114, %arg34 : i32
      %214 = tt.splat %213 : i32 -> tensor<64xi32>
      %215 = arith.addi %214, %52 : tensor<64xi32>
      %216 = tt.load %211 : tensor<16x64x!tt.ptr<f32>>
      %217 = tt.expand_dims %215 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %218 = arith.addi %217, %cst_1 : tensor<1x64xi32>
      %219 = tt.broadcast %218 : tensor<1x64xi32> -> tensor<128x64xi32>
      %220 = arith.subi %219, %125 : tensor<128x64xi32>
      %221 = arith.subi %220, %cst_2 : tensor<128x64xi32>
      %222 = arith.addi %221, %129 : tensor<128x64xi32>
      %223 = arith.addi %222, %133 : tensor<128x64xi32>
      %224 = tt.bitcast %221 : tensor<128x64xi32> -> tensor<128x64xi32>
      %225 = arith.cmpi ult, %224, %cst_2 : tensor<128x64xi32>
      %226 = arith.extsi %223 : tensor<128x64xi32> to tensor<128x64xi64>
      %227 = arith.addi %134, %226 : tensor<128x64xi64>
      %228 = "tts.create_ptr"(%arg30, %227) : (!tt.ptr<f32>, tensor<128x64xi64>) -> tensor<128x64x!tt.ptr<f32>>
      %229 = tt.load %228, %225 : tensor<128x64x!tt.ptr<f32>>
      %230 = arith.select %225, %229, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
      %231 = tt.dot %103, %216, %230, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
      %232 = tt.broadcast %217 : tensor<1x64xi32> -> tensor<128x64xi32>
      %233 = arith.subi %125, %232 : tensor<128x64xi32>
      %234 = arith.cmpi sge, %233, %cst_4 : tensor<128x64xi32>
      %235 = arith.cmpi slt, %233, %cst_2 : tensor<128x64xi32>
      %236 = arith.andi %234, %235 : tensor<128x64xi1>
      %237 = arith.mulf %231, %135 : tensor<128x64xf32>
      %238 = arith.select %236, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
      %239 = arith.addf %237, %238 : tensor<128x64xf32>
      %240 = "tt.reduce"(%239) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %260 = arith.maxnumf %arg40, %arg41 : f32
        tt.reduce.return %260 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %241 = arith.maxnumf %arg37, %240 : tensor<128xf32>
      %242 = tt.expand_dims %241 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %243 = tt.broadcast %242 : tensor<128x1xf32> -> tensor<128x64xf32>
      %244 = arith.subf %239, %243 : tensor<128x64xf32>
      %245 = math.exp2 %244 : tensor<128x64xf32>
      %246 = "tt.reduce"(%245) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %260 = arith.addf %arg40, %arg41 : f32
        tt.reduce.return %260 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %247 = arith.subf %arg37, %241 : tensor<128xf32>
      %248 = math.exp2 %247 : tensor<128xf32>
      %249 = arith.mulf %arg35, %248 : tensor<128xf32>
      %250 = arith.addf %249, %246 : tensor<128xf32>
      %251 = tt.expand_dims %248 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %252 = tt.broadcast %251 : tensor<128x1xf32> -> tensor<128x16xf32>
      %253 = arith.mulf %arg36, %252 : tensor<128x16xf32>
      %254 = tt.load %212 : tensor<64x16x!tt.ptr<f32>>
      %255 = tt.dot %245, %254, %253, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
      %256 = arith.extsi %137 : tensor<64x16xi32> to tensor<64x16xi64>
      %257 = arith.addi %arg38, %256 : tensor<64x16xi64>
      %258 = arith.extsi %138 : tensor<16x64xi32> to tensor<16x64xi64>
      %259 = arith.addi %arg39, %258 : tensor<16x64xi64>
      scf.yield %250, %255, %241, %257, %259 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %140 = arith.addi %110, %c2_i64 : i64
    %141 = "tts.create_ptr"(%arg17, %140) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %142 = arith.addi %110, %c3_i64 : i64
    %143 = "tts.create_ptr"(%arg17, %142) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %144 = tt.load %141 : !tt.ptr<i32>
    %145 = tt.load %143 : !tt.ptr<i32>
    %146 = arith.muli %144, %arg23 : i32
    %147 = tt.splat %146 : i32 -> tensor<16x64xi32>
    %148 = arith.extsi %147 : tensor<16x64xi32> to tensor<16x64xi64>
    %149 = arith.addi %91, %148 : tensor<16x64xi64>
    %150 = tt.splat %146 : i32 -> tensor<64x16xi32>
    %151 = arith.extsi %150 : tensor<64x16xi32> to tensor<64x16xi64>
    %152 = arith.addi %101, %151 : tensor<64x16xi64>
    %153 = arith.subi %145, %144 : i32
    %154 = tt.splat %102 : f32 -> tensor<128xf32>
    %155:5 = scf.for %arg34 = %c0_i32 to %153 step %c64_i32 iter_args(%arg35 = %139#0, %arg36 = %139#1, %arg37 = %139#2, %arg38 = %152, %arg39 = %149) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %211 = "tts.create_ptr"(%arg3, %arg39) : (!tt.ptr<f32>, tensor<16x64xi64>) -> tensor<16x64x!tt.ptr<f32>>
      %212 = "tts.create_ptr"(%arg4, %arg38) : (!tt.ptr<f32>, tensor<64x16xi64>) -> tensor<64x16x!tt.ptr<f32>>
      %213 = arith.addi %144, %arg34 : i32
      %214 = tt.splat %213 : i32 -> tensor<64xi32>
      %215 = arith.addi %214, %52 : tensor<64xi32>
      %216 = tt.load %211 : tensor<16x64x!tt.ptr<f32>>
      %217 = tt.expand_dims %215 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %218 = arith.addi %217, %cst_1 : tensor<1x64xi32>
      %219 = tt.broadcast %218 : tensor<1x64xi32> -> tensor<128x64xi32>
      %220 = arith.subi %219, %125 : tensor<128x64xi32>
      %221 = arith.subi %220, %cst_2 : tensor<128x64xi32>
      %222 = arith.addi %221, %129 : tensor<128x64xi32>
      %223 = arith.addi %222, %133 : tensor<128x64xi32>
      %224 = tt.bitcast %221 : tensor<128x64xi32> -> tensor<128x64xi32>
      %225 = arith.cmpi ult, %224, %cst_2 : tensor<128x64xi32>
      %226 = arith.extsi %223 : tensor<128x64xi32> to tensor<128x64xi64>
      %227 = arith.addi %134, %226 : tensor<128x64xi64>
      %228 = "tts.create_ptr"(%arg30, %227) : (!tt.ptr<f32>, tensor<128x64xi64>) -> tensor<128x64x!tt.ptr<f32>>
      %229 = tt.load %228, %225 : tensor<128x64x!tt.ptr<f32>>
      %230 = arith.select %225, %229, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
      %231 = tt.dot %103, %216, %230, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
      %232 = "tt.reduce"(%231) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %254 = arith.maxnumf %arg40, %arg41 : f32
        tt.reduce.return %254 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %233 = arith.mulf %232, %154 : tensor<128xf32>
      %234 = arith.maxnumf %arg37, %233 : tensor<128xf32>
      %235 = arith.mulf %231, %135 : tensor<128x64xf32>
      %236 = tt.expand_dims %234 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %237 = tt.broadcast %236 : tensor<128x1xf32> -> tensor<128x64xf32>
      %238 = arith.subf %235, %237 : tensor<128x64xf32>
      %239 = math.exp2 %238 : tensor<128x64xf32>
      %240 = "tt.reduce"(%239) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %254 = arith.addf %arg40, %arg41 : f32
        tt.reduce.return %254 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %241 = arith.subf %arg37, %234 : tensor<128xf32>
      %242 = math.exp2 %241 : tensor<128xf32>
      %243 = arith.mulf %arg35, %242 : tensor<128xf32>
      %244 = arith.addf %243, %240 : tensor<128xf32>
      %245 = tt.expand_dims %242 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %246 = tt.broadcast %245 : tensor<128x1xf32> -> tensor<128x16xf32>
      %247 = arith.mulf %arg36, %246 : tensor<128x16xf32>
      %248 = tt.load %212 : tensor<64x16x!tt.ptr<f32>>
      %249 = tt.dot %239, %248, %247, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
      %250 = arith.extsi %137 : tensor<64x16xi32> to tensor<64x16xi64>
      %251 = arith.addi %arg38, %250 : tensor<64x16xi64>
      %252 = arith.extsi %138 : tensor<16x64xi32> to tensor<16x64xi64>
      %253 = arith.addi %arg39, %252 : tensor<16x64xi64>
      scf.yield %244, %249, %234, %251, %253 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %156 = arith.addi %110, %c4_i64 : i64
    %157 = "tts.create_ptr"(%arg17, %156) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %158 = arith.addi %110, %c5_i64 : i64
    %159 = "tts.create_ptr"(%arg17, %158) : (!tt.ptr<i32>, i64) -> !tt.ptr<i32>
    %160 = tt.load %157 : !tt.ptr<i32>
    %161 = tt.load %159 : !tt.ptr<i32>
    %162 = arith.muli %160, %arg23 : i32
    %163 = tt.splat %162 : i32 -> tensor<16x64xi32>
    %164 = arith.extsi %163 : tensor<16x64xi32> to tensor<16x64xi64>
    %165 = arith.addi %91, %164 : tensor<16x64xi64>
    %166 = tt.splat %162 : i32 -> tensor<64x16xi32>
    %167 = arith.extsi %166 : tensor<64x16xi32> to tensor<64x16xi64>
    %168 = arith.addi %101, %167 : tensor<64x16xi64>
    %169 = arith.subi %161, %160 : i32
    %170 = tt.splat %40 : i32 -> tensor<64xi32>
    %171:5 = scf.for %arg34 = %c0_i32 to %169 step %c64_i32 iter_args(%arg35 = %155#0, %arg36 = %155#1, %arg37 = %155#2, %arg38 = %168, %arg39 = %165) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %211 = "tts.create_ptr"(%arg3, %arg39) : (!tt.ptr<f32>, tensor<16x64xi64>) -> tensor<16x64x!tt.ptr<f32>>
      %212 = "tts.create_ptr"(%arg4, %arg38) : (!tt.ptr<f32>, tensor<64x16xi64>) -> tensor<64x16x!tt.ptr<f32>>
      %213 = arith.addi %160, %arg34 : i32
      %214 = tt.splat %213 : i32 -> tensor<64xi32>
      %215 = arith.addi %214, %52 : tensor<64xi32>
      %216 = arith.cmpi slt, %215, %170 : tensor<64xi32>
      %217 = tt.expand_dims %216 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %218 = tt.broadcast %217 : tensor<1x64xi1> -> tensor<16x64xi1>
      %219 = tt.load %211, %218 : tensor<16x64x!tt.ptr<f32>>
      %220 = tt.expand_dims %215 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %221 = arith.addi %220, %cst_1 : tensor<1x64xi32>
      %222 = tt.broadcast %221 : tensor<1x64xi32> -> tensor<128x64xi32>
      %223 = arith.subi %222, %125 : tensor<128x64xi32>
      %224 = arith.subi %223, %cst_2 : tensor<128x64xi32>
      %225 = arith.addi %224, %129 : tensor<128x64xi32>
      %226 = arith.addi %225, %133 : tensor<128x64xi32>
      %227 = tt.bitcast %224 : tensor<128x64xi32> -> tensor<128x64xi32>
      %228 = arith.cmpi ult, %227, %cst_2 : tensor<128x64xi32>
      %229 = arith.extsi %226 : tensor<128x64xi32> to tensor<128x64xi64>
      %230 = arith.addi %134, %229 : tensor<128x64xi64>
      %231 = "tts.create_ptr"(%arg30, %230) : (!tt.ptr<f32>, tensor<128x64xi64>) -> tensor<128x64x!tt.ptr<f32>>
      %232 = tt.load %231, %228 : tensor<128x64x!tt.ptr<f32>>
      %233 = arith.select %228, %232, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
      %234 = tt.dot %103, %219, %233, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
      %235 = tt.broadcast %220 : tensor<1x64xi32> -> tensor<128x64xi32>
      %236 = arith.subi %125, %235 : tensor<128x64xi32>
      %237 = arith.cmpi sge, %236, %cst_4 : tensor<128x64xi32>
      %238 = arith.cmpi slt, %236, %cst_2 : tensor<128x64xi32>
      %239 = arith.andi %237, %238 : tensor<128x64xi1>
      %240 = tt.broadcast %217 : tensor<1x64xi1> -> tensor<128x64xi1>
      %241 = arith.andi %239, %240 : tensor<128x64xi1>
      %242 = arith.mulf %234, %135 : tensor<128x64xf32>
      %243 = arith.select %241, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
      %244 = arith.addf %242, %243 : tensor<128x64xf32>
      %245 = "tt.reduce"(%244) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %267 = arith.maxnumf %arg40, %arg41 : f32
        tt.reduce.return %267 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %246 = arith.maxnumf %arg37, %245 : tensor<128xf32>
      %247 = tt.expand_dims %246 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %248 = tt.broadcast %247 : tensor<128x1xf32> -> tensor<128x64xf32>
      %249 = arith.subf %244, %248 : tensor<128x64xf32>
      %250 = math.exp2 %249 : tensor<128x64xf32>
      %251 = "tt.reduce"(%250) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %267 = arith.addf %arg40, %arg41 : f32
        tt.reduce.return %267 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %252 = arith.subf %arg37, %246 : tensor<128xf32>
      %253 = math.exp2 %252 : tensor<128xf32>
      %254 = arith.mulf %arg35, %253 : tensor<128xf32>
      %255 = arith.addf %254, %251 : tensor<128xf32>
      %256 = tt.expand_dims %253 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %257 = tt.broadcast %256 : tensor<128x1xf32> -> tensor<128x16xf32>
      %258 = arith.mulf %arg36, %257 : tensor<128x16xf32>
      %259 = tt.expand_dims %216 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
      %260 = tt.broadcast %259 : tensor<64x1xi1> -> tensor<64x16xi1>
      %261 = tt.load %212, %260, %cst_0 : tensor<64x16x!tt.ptr<f32>>
      %262 = tt.dot %250, %261, %258, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
      %263 = arith.extsi %137 : tensor<64x16xi32> to tensor<64x16xi64>
      %264 = arith.addi %arg38, %263 : tensor<64x16xi64>
      %265 = arith.extsi %138 : tensor<16x64xi32> to tensor<16x64xi64>
      %266 = arith.addi %arg39, %265 : tensor<16x64xi64>
      scf.yield %255, %262, %246, %264, %266 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %172 = math.log2 %171#0 : tensor<128xf32>
    %173 = arith.addf %171#2, %172 : tensor<128xf32>
    %174 = tt.expand_dims %171#0 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %175 = tt.broadcast %174 : tensor<128x1xf32> -> tensor<128x16xf32>
    %176 = arith.divf %171#1, %175 : tensor<128x16xf32>
    %177 = arith.index_cast %arg20 : i32 to index
    %178 = arith.muli %25, %177 : index
    %179 = arith.addi %17, %178 : index
    %180 = tts.make_tptr %arg6 to sizes: [128, 16], strides: [%177, 1], offsets: [%179, 0], shape: [0, 0], order: [] : <f32> to tensor<128x16x!tt.ptr<f32>>
    %181 = arith.addi %21, %25 : index
    %182 = tts.make_tptr %arg7 to sizes: [128], strides: [1], offsets: [%181], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    %183 = arith.addi %25, %c128 : index
    %184 = arith.index_cast %32 : i32 to index
    %185 = arith.minsi %183, %184 : index
    %186 = arith.subi %185, %25 : index
    %187 = "tts.load"(%182, %186) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, index) -> tensor<128xf32>
    %188 = tt.expand_dims %34 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %189 = tt.broadcast %188 : tensor<128x1xi1> -> tensor<128x16xi1>
    %190 = "tts.load"(%180, %186) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 16>}> : (tensor<128x16x!tt.ptr<f32>>, index) -> tensor<128x16xf32>
    %191 = arith.maxnumf %187, %173 : tensor<128xf32>
    %192 = arith.subf %173, %191 : tensor<128xf32>
    %193 = math.exp2 %192 : tensor<128xf32>
    %194 = arith.subf %187, %191 : tensor<128xf32>
    %195 = math.exp2 %194 : tensor<128xf32>
    %196 = arith.addf %193, %195 : tensor<128xf32>
    %197 = math.log2 %196 : tensor<128xf32>
    %198 = arith.addf %191, %197 : tensor<128xf32>
    %199 = arith.subf %173, %198 : tensor<128xf32>
    %200 = math.exp2 %199 : tensor<128xf32>
    %201 = tt.expand_dims %200 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %202 = tt.broadcast %201 : tensor<128x1xf32> -> tensor<128x16xf32>
    %203 = arith.mulf %176, %202 : tensor<128x16xf32>
    %204 = arith.subf %187, %198 : tensor<128xf32>
    %205 = math.exp2 %204 : tensor<128xf32>
    %206 = tt.expand_dims %205 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %207 = tt.broadcast %206 : tensor<128x1xf32> -> tensor<128x16xf32>
    %208 = arith.mulf %190, %207 : tensor<128x16xf32>
    %209 = arith.addf %203, %208 : tensor<128x16xf32>
    %210 = tts.make_tptr %arg1 to sizes: [128], strides: [1], offsets: [%181], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    "tts.store"(%210, %198, %186) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>, index) -> ()
    tt.store %80, %209, %189 : tensor<128x16x!tt.ptr<f32>>
    tt.return
  }
}

