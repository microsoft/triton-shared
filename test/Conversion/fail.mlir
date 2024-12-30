/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:30:10: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %4 = tt.load %3 : !tt.ptr<i32>
         ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:30:10: note: see current operation: %5 = tt.load %4 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:30:10: remark: PtrAnalysis: Failed to rewrite LoadOp
    %4 = tt.load %3 : !tt.ptr<i32>
         ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:30:10: note: see current operation: %5 = tt.load %4 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:38:10: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %8 = tt.load %7 : !tt.ptr<i32>
         ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:38:10: note: see current operation: %10 = tt.load %9 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:38:10: remark: PtrAnalysis: Failed to rewrite LoadOp
    %8 = tt.load %7 : !tt.ptr<i32>
         ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:38:10: note: see current operation: %10 = tt.load %9 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:58:11: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %28 = tt.load %27 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:58:11: note: see current operation: %34 = tt.load %33 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:58:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %28 = tt.load %27 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:58:11: note: see current operation: %34 = tt.load %33 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:66:11: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %36 = tt.load %35 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:66:11: note: see current operation: %43 = tt.load %42 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:66:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %36 = tt.load %35 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:66:11: note: see current operation: %43 = tt.load %42 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:68:11: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %38 = tt.load %37 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:68:11: note: see current operation: %46 = tt.load %45 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:68:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %38 = tt.load %37 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:68:11: note: see current operation: %46 = tt.load %45 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:76:11: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %46 = tt.load %45 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:76:11: note: see current operation: %57 = tt.load %56 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:76:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %46 = tt.load %45 : !tt.ptr<i32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:76:11: note: see current operation: %57 = tt.load %56 : !tt.ptr<i32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%41 = arith.select %40, %35, %34 : tensor<128xi1>, tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:94:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %64 = tt.addptr %63, %62 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:94:11: note: see current operation: %78 = tt.addptr %77, %76 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%41 = arith.select %40, %35, %34 : tensor<128xi1>, tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:98:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %68 = tt.addptr %66, %67 : tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:98:11: note: see current operation: %82 = tt.addptr %80, %81 : tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%41 = arith.select %40, %35, %34 : tensor<128xi1>, tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:100:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %70 = tt.addptr %69, %62 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:100:11: note: see current operation: %84 = tt.addptr %83, %76 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%41 = arith.select %40, %35, %34 : tensor<128xi1>, tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:102:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %72 = tt.addptr %71, %67 : tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:102:11: note: see current operation: %86 = tt.addptr %85, %81 : tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%64 = "tts.load"(%62) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:107:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %77 = tt.addptr %76, %75 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:107:11: note: see current operation: %91 = tt.addptr %90, %89 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%64 = "tts.load"(%62) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:111:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %81 = tt.addptr %79, %80 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:111:11: note: see current operation: %95 = tt.addptr %93, %94 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%64 = "tts.load"(%62) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:116:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %86 = tt.addptr %85, %84 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:116:11: note: see current operation: %100 = tt.addptr %99, %98 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%64 = "tts.load"(%62) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:119:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %89 = tt.addptr %87, %88 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:119:11: note: see current operation: %103 = tt.addptr %101, %102 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:121:11: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
    %91 = tt.load %68 : tensor<128x16x!tt.ptr<f32>>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:121:11: note: see current operation: %105 = tt.load %82 : tensor<128x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:121:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %91 = tt.load %68 : tensor<128x16x!tt.ptr<f32>>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:121:11: note: see current operation: %105 = tt.load %82 : tensor<128x16x!tt.ptr<f32>>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%42 = arith.divsi %41, %cst_9 : tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:124:11: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %94 = tt.addptr %93, %33 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:124:11: note: see current operation: %109 = tt.addptr %108, %42 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:125:11: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
    %95 = tt.load %94 : tensor<128x!tt.ptr<i32>>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:125:11: note: see current operation: %110 = tt.load %109 : tensor<128x!tt.ptr<i32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:125:11: remark: PtrAnalysis: Failed to rewrite LoadOp
    %95 = tt.load %94 : tensor<128x!tt.ptr<i32>>
          ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:125:11: note: see current operation: %110 = tt.load %109 : tensor<128x!tt.ptr<i32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:130:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %100 = tt.load %98 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:130:12: note: see current operation: %120 = tt.load %117 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:130:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %100 = tt.load %98 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:130:12: note: see current operation: %120 = tt.load %117 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:131:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %101 = tt.load %99 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:131:12: note: see current operation: %121 = tt.load %119 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:131:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %101 = tt.load %99 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:131:12: note: see current operation: %121 = tt.load %119 : !tt.ptr<i32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%66 = "tts.load"(%64) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %104 = tt.addptr %81, %103 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: note: see current operation: %124 = tt.addptr %98, %123 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%66 = "tts.load"(%64) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %106 = tt.addptr %89, %105 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: note: see current operation: %126 = tt.addptr %106, %125 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%66 = "tts.load"(%64) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %106 = tt.addptr %89, %105 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: note: see current operation: %structured, %offsets:2, %strides:2 = "tts.get_structured_state"(%126) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%66 = "tts.load"(%64) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %104 = tt.addptr %81, %103 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: note: see current operation: %structured_10, %offsets_11:2, %strides_12:2 = "tts.get_structured_state"(%124) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:153:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 3
    %123:5 = scf.for %arg34 = %c0_i32 to %107 step %c64_i32 iter_args(%arg35 = %cst_7, %arg36 = %cst_6, %arg37 = %cst, %arg38 = %106, %arg39 = %104) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:153:14: note: see current operation: 
%143:13 = scf.for %arg34 = %c0_i32 to %127 step %c64_i32 iter_args(%arg35 = %cst_7, %arg36 = %cst_6, %arg37 = %cst, %arg38 = %structured, %arg39 = %offsets#0, %arg40 = %offsets#1, %arg41 = %strides#0, %arg42 = %strides#1, %arg43 = %structured_10, %arg44 = %offsets_11#0, %arg45 = %offsets_11#1, %arg46 = %strides_12#0, %arg47 = %strides_12#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %238 = arith.addi %120, %arg34 : i32
  %239 = tt.splat %238 : i32 -> tensor<64xi32>
  %240 = arith.addi %239, %62 : tensor<64xi32>
  %241 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
  %242 = tt.expand_dims %240 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %243 = arith.addi %242, %cst_1 : tensor<1x64xi32>
  %244 = tt.broadcast %243 : tensor<1x64xi32> -> tensor<128x64xi32>
  %245 = arith.subi %244, %129 : tensor<128x64xi32>
  %246 = arith.subi %245, %cst_2 : tensor<128x64xi32>
  %247 = arith.addi %246, %133 : tensor<128x64xi32>
  %248 = arith.addi %247, %137 : tensor<128x64xi32>
  %249 = tt.bitcast %246 : tensor<128x64xi32> -> tensor<128x64xi32>
  %250 = arith.cmpi ult, %249, %cst_2 : tensor<128x64xi32>
  %251 = tt.addptr %138, %248 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %252 = tt.load %251, %250 : tensor<128x64x!tt.ptr<f32>>
  %253 = arith.select %250, %252, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %254 = tt.dot %108, %241, %253, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %255 = tt.broadcast %242 : tensor<1x64xi32> -> tensor<128x64xi32>
  %256 = arith.subi %129, %255 : tensor<128x64xi32>
  %257 = arith.cmpi sge, %256, %cst_4 : tensor<128x64xi32>
  %258 = arith.cmpi slt, %256, %cst_2 : tensor<128x64xi32>
  %259 = arith.andi %257, %258 : tensor<128x64xi1>
  %260 = arith.mulf %254, %139 : tensor<128x64xf32>
  %261 = arith.select %259, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
  %262 = arith.addf %260, %261 : tensor<128x64xf32>
  %263 = "tt.reduce"(%262) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %281 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %281 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %264 = arith.maxnumf %arg37, %263 : tensor<128xf32>
  %265 = tt.expand_dims %264 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %266 = tt.broadcast %265 : tensor<128x1xf32> -> tensor<128x64xf32>
  %267 = arith.subf %262, %266 : tensor<128x64xf32>
  %268 = math.exp2 %267 : tensor<128x64xf32>
  %269 = "tt.reduce"(%268) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %281 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %281 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %270 = arith.subf %arg37, %264 : tensor<128xf32>
  %271 = math.exp2 %270 : tensor<128xf32>
  %272 = arith.mulf %arg35, %271 : tensor<128xf32>
  %273 = arith.addf %272, %269 : tensor<128xf32>
  %274 = tt.expand_dims %271 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %275 = tt.broadcast %274 : tensor<128x1xf32> -> tensor<128x16xf32>
  %276 = arith.mulf %arg36, %275 : tensor<128x16xf32>
  %277 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
  %278 = tt.dot %268, %277, %276, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %279 = tt.addptr %arg38, %141 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %280 = tt.addptr %arg43, %142 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%279) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%280) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %273, %278, %264, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:153:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 8
    %123:5 = scf.for %arg34 = %c0_i32 to %107 step %c64_i32 iter_args(%arg35 = %cst_7, %arg36 = %cst_6, %arg37 = %cst, %arg38 = %106, %arg39 = %104) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:153:14: note: see current operation: 
%143:13 = scf.for %arg34 = %c0_i32 to %127 step %c64_i32 iter_args(%arg35 = %cst_7, %arg36 = %cst_6, %arg37 = %cst, %arg38 = %structured, %arg39 = %offsets#0, %arg40 = %offsets#1, %arg41 = %strides#0, %arg42 = %strides#1, %arg43 = %structured_10, %arg44 = %offsets_11#0, %arg45 = %offsets_11#1, %arg46 = %strides_12#0, %arg47 = %strides_12#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %238 = arith.addi %120, %arg34 : i32
  %239 = tt.splat %238 : i32 -> tensor<64xi32>
  %240 = arith.addi %239, %62 : tensor<64xi32>
  %241 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
  %242 = tt.expand_dims %240 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %243 = arith.addi %242, %cst_1 : tensor<1x64xi32>
  %244 = tt.broadcast %243 : tensor<1x64xi32> -> tensor<128x64xi32>
  %245 = arith.subi %244, %129 : tensor<128x64xi32>
  %246 = arith.subi %245, %cst_2 : tensor<128x64xi32>
  %247 = arith.addi %246, %133 : tensor<128x64xi32>
  %248 = arith.addi %247, %137 : tensor<128x64xi32>
  %249 = tt.bitcast %246 : tensor<128x64xi32> -> tensor<128x64xi32>
  %250 = arith.cmpi ult, %249, %cst_2 : tensor<128x64xi32>
  %251 = tt.addptr %138, %248 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %252 = tt.load %251, %250 : tensor<128x64x!tt.ptr<f32>>
  %253 = arith.select %250, %252, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %254 = tt.dot %108, %241, %253, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %255 = tt.broadcast %242 : tensor<1x64xi32> -> tensor<128x64xi32>
  %256 = arith.subi %129, %255 : tensor<128x64xi32>
  %257 = arith.cmpi sge, %256, %cst_4 : tensor<128x64xi32>
  %258 = arith.cmpi slt, %256, %cst_2 : tensor<128x64xi32>
  %259 = arith.andi %257, %258 : tensor<128x64xi1>
  %260 = arith.mulf %254, %139 : tensor<128x64xf32>
  %261 = arith.select %259, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
  %262 = arith.addf %260, %261 : tensor<128x64xf32>
  %263 = "tt.reduce"(%262) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %281 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %281 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %264 = arith.maxnumf %arg37, %263 : tensor<128xf32>
  %265 = tt.expand_dims %264 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %266 = tt.broadcast %265 : tensor<128x1xf32> -> tensor<128x64xf32>
  %267 = arith.subf %262, %266 : tensor<128x64xf32>
  %268 = math.exp2 %267 : tensor<128x64xf32>
  %269 = "tt.reduce"(%268) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %281 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %281 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %270 = arith.subf %arg37, %264 : tensor<128xf32>
  %271 = math.exp2 %270 : tensor<128xf32>
  %272 = arith.mulf %arg35, %271 : tensor<128xf32>
  %273 = arith.addf %272, %269 : tensor<128xf32>
  %274 = tt.expand_dims %271 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %275 = tt.broadcast %274 : tensor<128x1xf32> -> tensor<128x16xf32>
  %276 = arith.mulf %arg36, %275 : tensor<128x16xf32>
  %277 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
  %278 = tt.dot %268, %277, %276, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %279 = tt.addptr %arg38, %141 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %280 = tt.addptr %arg43, %142 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%279) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%280) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %273, %278, %264, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:157:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %221 = tt.load %arg39 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:157:14: note: see current operation: %241 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:157:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %221 = tt.load %arg39 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:157:14: note: see current operation: %241 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%246 = arith.subi %245, %cst_2 : tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:167:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %231 = tt.addptr %118, %228 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:167:14: note: see current operation: %251 = tt.addptr %138, %248 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:168:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %232 = tt.load %231, %230 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:168:14: note: see current operation: %252 = tt.load %251, %250 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:168:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %232 = tt.load %231, %230 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:168:14: note: see current operation: %252 = tt.load %251, %250 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:201:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %257 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:201:14: note: see current operation: %277 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:201:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %257 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:201:14: note: see current operation: %277 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %259 = tt.addptr %arg38, %121 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: note: see current operation: %279 = tt.addptr %arg38, %141 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %260 = tt.addptr %arg39, %122 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: note: see current operation: %280 = tt.addptr %arg43, %142 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %259 = tt.addptr %arg38, %121 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: note: see current operation: %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%279) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %260 = tt.addptr %arg39, %122 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: note: see current operation: %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%280) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:209:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %126 = tt.load %124 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:209:12: note: see current operation: %150 = tt.load %147 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:209:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %126 = tt.load %124 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:209:12: note: see current operation: %150 = tt.load %147 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:210:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %127 = tt.load %125 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:210:12: note: see current operation: %151 = tt.load %149 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:210:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %127 = tt.load %125 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:210:12: note: see current operation: %151 = tt.load %149 : !tt.ptr<i32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%68 = "tts.load"(%66) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %130 = tt.addptr %81, %129 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: note: see current operation: %154 = tt.addptr %100, %153 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%68 = "tts.load"(%66) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %132 = tt.addptr %89, %131 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: note: see current operation: %156 = tt.addptr %108, %155 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%68 = "tts.load"(%66) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %132 = tt.addptr %89, %131 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: note: see current operation: %structured_13, %offsets_14:2, %strides_15:2 = "tts.get_structured_state"(%156) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%68 = "tts.load"(%66) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %130 = tt.addptr %81, %129 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: note: see current operation: %structured_16, %offsets_17:2, %strides_18:2 = "tts.get_structured_state"(%154) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:233:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 3
    %150:5 = scf.for %arg34 = %c0_i32 to %133 step %c64_i32 iter_args(%arg35 = %123#0, %arg36 = %123#1, %arg37 = %123#2, %arg38 = %132, %arg39 = %130) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:233:14: note: see current operation: 
%174:13 = scf.for %arg34 = %c0_i32 to %157 step %c64_i32 iter_args(%arg35 = %145#0, %arg36 = %145#1, %arg37 = %145#2, %arg38 = %structured_13, %arg39 = %offsets_14#0, %arg40 = %offsets_14#1, %arg41 = %strides_15#0, %arg42 = %strides_15#1, %arg43 = %structured_16, %arg44 = %offsets_17#0, %arg45 = %offsets_17#1, %arg46 = %strides_18#0, %arg47 = %strides_18#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %242 = arith.addi %150, %arg34 : i32
  %243 = tt.splat %242 : i32 -> tensor<64xi32>
  %244 = arith.addi %243, %64 : tensor<64xi32>
  %245 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
  %246 = tt.expand_dims %244 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %247 = arith.addi %246, %cst_1 : tensor<1x64xi32>
  %248 = tt.broadcast %247 : tensor<1x64xi32> -> tensor<128x64xi32>
  %249 = arith.subi %248, %159 : tensor<128x64xi32>
  %250 = arith.subi %249, %cst_2 : tensor<128x64xi32>
  %251 = arith.addi %250, %163 : tensor<128x64xi32>
  %252 = arith.addi %251, %167 : tensor<128x64xi32>
  %253 = tt.bitcast %250 : tensor<128x64xi32> -> tensor<128x64xi32>
  %254 = arith.cmpi ult, %253, %cst_2 : tensor<128x64xi32>
  %255 = tt.addptr %168, %252 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %256 = tt.load %255, %254 : tensor<128x64x!tt.ptr<f32>>
  %257 = arith.select %254, %256, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %258 = tt.dot %110, %245, %257, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %259 = "tt.reduce"(%258) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %279 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %279 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %260 = arith.mulf %259, %169 : tensor<128xf32>
  %261 = arith.maxnumf %arg37, %260 : tensor<128xf32>
  %262 = arith.mulf %258, %170 : tensor<128x64xf32>
  %263 = tt.expand_dims %261 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %264 = tt.broadcast %263 : tensor<128x1xf32> -> tensor<128x64xf32>
  %265 = arith.subf %262, %264 : tensor<128x64xf32>
  %266 = math.exp2 %265 : tensor<128x64xf32>
  %267 = "tt.reduce"(%266) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %279 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %279 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %268 = arith.subf %arg37, %261 : tensor<128xf32>
  %269 = math.exp2 %268 : tensor<128xf32>
  %270 = arith.mulf %arg35, %269 : tensor<128xf32>
  %271 = arith.addf %270, %267 : tensor<128xf32>
  %272 = tt.expand_dims %269 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %273 = tt.broadcast %272 : tensor<128x1xf32> -> tensor<128x16xf32>
  %274 = arith.mulf %arg36, %273 : tensor<128x16xf32>
  %275 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
  %276 = tt.dot %266, %275, %274, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %277 = tt.addptr %arg38, %172 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %278 = tt.addptr %arg43, %173 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%277) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%278) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %271, %276, %261, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:233:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 8
    %150:5 = scf.for %arg34 = %c0_i32 to %133 step %c64_i32 iter_args(%arg35 = %123#0, %arg36 = %123#1, %arg37 = %123#2, %arg38 = %132, %arg39 = %130) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:233:14: note: see current operation: 
%174:13 = scf.for %arg34 = %c0_i32 to %157 step %c64_i32 iter_args(%arg35 = %145#0, %arg36 = %145#1, %arg37 = %145#2, %arg38 = %structured_13, %arg39 = %offsets_14#0, %arg40 = %offsets_14#1, %arg41 = %strides_15#0, %arg42 = %strides_15#1, %arg43 = %structured_16, %arg44 = %offsets_17#0, %arg45 = %offsets_17#1, %arg46 = %strides_18#0, %arg47 = %strides_18#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %242 = arith.addi %150, %arg34 : i32
  %243 = tt.splat %242 : i32 -> tensor<64xi32>
  %244 = arith.addi %243, %64 : tensor<64xi32>
  %245 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
  %246 = tt.expand_dims %244 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %247 = arith.addi %246, %cst_1 : tensor<1x64xi32>
  %248 = tt.broadcast %247 : tensor<1x64xi32> -> tensor<128x64xi32>
  %249 = arith.subi %248, %159 : tensor<128x64xi32>
  %250 = arith.subi %249, %cst_2 : tensor<128x64xi32>
  %251 = arith.addi %250, %163 : tensor<128x64xi32>
  %252 = arith.addi %251, %167 : tensor<128x64xi32>
  %253 = tt.bitcast %250 : tensor<128x64xi32> -> tensor<128x64xi32>
  %254 = arith.cmpi ult, %253, %cst_2 : tensor<128x64xi32>
  %255 = tt.addptr %168, %252 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %256 = tt.load %255, %254 : tensor<128x64x!tt.ptr<f32>>
  %257 = arith.select %254, %256, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %258 = tt.dot %110, %245, %257, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %259 = "tt.reduce"(%258) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %279 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %279 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %260 = arith.mulf %259, %169 : tensor<128xf32>
  %261 = arith.maxnumf %arg37, %260 : tensor<128xf32>
  %262 = arith.mulf %258, %170 : tensor<128x64xf32>
  %263 = tt.expand_dims %261 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %264 = tt.broadcast %263 : tensor<128x1xf32> -> tensor<128x64xf32>
  %265 = arith.subf %262, %264 : tensor<128x64xf32>
  %266 = math.exp2 %265 : tensor<128x64xf32>
  %267 = "tt.reduce"(%266) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %279 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %279 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %268 = arith.subf %arg37, %261 : tensor<128xf32>
  %269 = math.exp2 %268 : tensor<128xf32>
  %270 = arith.mulf %arg35, %269 : tensor<128xf32>
  %271 = arith.addf %270, %267 : tensor<128xf32>
  %272 = tt.expand_dims %269 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %273 = tt.broadcast %272 : tensor<128x1xf32> -> tensor<128x16xf32>
  %274 = arith.mulf %arg36, %273 : tensor<128x16xf32>
  %275 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
  %276 = tt.dot %266, %275, %274, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %277 = tt.addptr %arg38, %172 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %278 = tt.addptr %arg43, %173 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%277) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%278) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %271, %276, %261, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:237:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %221 = tt.load %arg39 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:237:14: note: see current operation: %245 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:237:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %221 = tt.load %arg39 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:237:14: note: see current operation: %245 = tt.load %arg43 : tensor<16x64x!tt.ptr<f32>>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%250 = arith.subi %249, %cst_2 : tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:247:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %231 = tt.addptr %144, %228 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:247:14: note: see current operation: %255 = tt.addptr %168, %252 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:248:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %232 = tt.load %231, %230 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:248:14: note: see current operation: %256 = tt.load %255, %254 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:248:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %232 = tt.load %231, %230 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:248:14: note: see current operation: %256 = tt.load %255, %254 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:275:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %251 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:275:14: note: see current operation: %275 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:275:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %251 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:275:14: note: see current operation: %275 = tt.load %arg38 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %253 = tt.addptr %arg38, %148 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: note: see current operation: %277 = tt.addptr %arg38, %172 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %254 = tt.addptr %arg39, %149 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: note: see current operation: %278 = tt.addptr %arg43, %173 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %253 = tt.addptr %arg38, %148 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: note: see current operation: %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%277) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %254 = tt.addptr %arg39, %149 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: note: see current operation: %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%278) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:283:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %153 = tt.load %151 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:283:12: note: see current operation: %181 = tt.load %178 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:283:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %153 = tt.load %151 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:283:12: note: see current operation: %181 = tt.load %178 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:284:12: remark: PtrAnalysis: scalar loadOp will not be rewritten
    %154 = tt.load %152 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:284:12: note: see current operation: %182 = tt.load %180 : !tt.ptr<i32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:284:12: remark: PtrAnalysis: Failed to rewrite LoadOp
    %154 = tt.load %152 : !tt.ptr<i32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:284:12: note: see current operation: %182 = tt.load %180 : !tt.ptr<i32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%70 = "tts.load"(%68) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %157 = tt.addptr %81, %156 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: note: see current operation: %185 = tt.addptr %102, %184 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%70 = "tts.load"(%68) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: remark: PtrAnalysis: Failed to rewrite AddPtrOp
    %159 = tt.addptr %89, %158 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: note: see current operation: %187 = tt.addptr %110, %186 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%70 = "tts.load"(%68) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %159 = tt.addptr %89, %158 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: note: see current operation: %structured_19, %offsets_20:2, %strides_21:2 = "tts.get_structured_state"(%187) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%70 = "tts.load"(%68) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
    %157 = tt.addptr %81, %156 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: note: see current operation: %structured_22, %offsets_23:2, %strides_24:2 = "tts.get_structured_state"(%185) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:307:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 3
    %177:5 = scf.for %arg34 = %c0_i32 to %160 step %c64_i32 iter_args(%arg35 = %150#0, %arg36 = %150#1, %arg37 = %150#2, %arg38 = %159, %arg39 = %157) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:307:14: note: see current operation: 
%205:13 = scf.for %arg34 = %c0_i32 to %188 step %c64_i32 iter_args(%arg35 = %176#0, %arg36 = %176#1, %arg37 = %176#2, %arg38 = %structured_19, %arg39 = %offsets_20#0, %arg40 = %offsets_20#1, %arg41 = %strides_21#0, %arg42 = %strides_21#1, %arg43 = %structured_22, %arg44 = %offsets_23#0, %arg45 = %offsets_23#1, %arg46 = %strides_24#0, %arg47 = %strides_24#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %246 = arith.addi %181, %arg34 : i32
  %247 = tt.splat %246 : i32 -> tensor<64xi32>
  %248 = arith.addi %247, %66 : tensor<64xi32>
  %249 = arith.cmpi slt, %248, %189 : tensor<64xi32>
  %250 = tt.expand_dims %249 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  %251 = tt.broadcast %250 : tensor<1x64xi1> -> tensor<16x64xi1>
  %252 = tt.load %arg43, %251 : tensor<16x64x!tt.ptr<f32>>
  %253 = tt.expand_dims %248 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %254 = arith.addi %253, %cst_1 : tensor<1x64xi32>
  %255 = tt.broadcast %254 : tensor<1x64xi32> -> tensor<128x64xi32>
  %256 = arith.subi %255, %191 : tensor<128x64xi32>
  %257 = arith.subi %256, %cst_2 : tensor<128x64xi32>
  %258 = arith.addi %257, %195 : tensor<128x64xi32>
  %259 = arith.addi %258, %199 : tensor<128x64xi32>
  %260 = tt.bitcast %257 : tensor<128x64xi32> -> tensor<128x64xi32>
  %261 = arith.cmpi ult, %260, %cst_2 : tensor<128x64xi32>
  %262 = tt.addptr %200, %259 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %263 = tt.load %262, %261 : tensor<128x64x!tt.ptr<f32>>
  %264 = arith.select %261, %263, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %265 = tt.dot %112, %252, %264, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %266 = tt.broadcast %253 : tensor<1x64xi32> -> tensor<128x64xi32>
  %267 = arith.subi %191, %266 : tensor<128x64xi32>
  %268 = arith.cmpi sge, %267, %cst_4 : tensor<128x64xi32>
  %269 = arith.cmpi slt, %267, %cst_2 : tensor<128x64xi32>
  %270 = arith.andi %268, %269 : tensor<128x64xi1>
  %271 = tt.broadcast %250 : tensor<1x64xi1> -> tensor<128x64xi1>
  %272 = arith.andi %270, %271 : tensor<128x64xi1>
  %273 = arith.mulf %265, %201 : tensor<128x64xf32>
  %274 = arith.select %272, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
  %275 = arith.addf %273, %274 : tensor<128x64xf32>
  %276 = "tt.reduce"(%275) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %296 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %296 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %277 = arith.maxnumf %arg37, %276 : tensor<128xf32>
  %278 = tt.expand_dims %277 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %279 = tt.broadcast %278 : tensor<128x1xf32> -> tensor<128x64xf32>
  %280 = arith.subf %275, %279 : tensor<128x64xf32>
  %281 = math.exp2 %280 : tensor<128x64xf32>
  %282 = "tt.reduce"(%281) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %296 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %296 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %283 = arith.subf %arg37, %277 : tensor<128xf32>
  %284 = math.exp2 %283 : tensor<128xf32>
  %285 = arith.mulf %arg35, %284 : tensor<128xf32>
  %286 = arith.addf %285, %282 : tensor<128xf32>
  %287 = tt.expand_dims %284 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %288 = tt.broadcast %287 : tensor<128x1xf32> -> tensor<128x16xf32>
  %289 = arith.mulf %arg36, %288 : tensor<128x16xf32>
  %290 = tt.expand_dims %249 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  %291 = tt.broadcast %290 : tensor<64x1xi1> -> tensor<64x16xi1>
  %292 = tt.load %arg38, %291, %cst_0 : tensor<64x16x!tt.ptr<f32>>
  %293 = tt.dot %281, %292, %289, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %294 = tt.addptr %arg38, %203 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %295 = tt.addptr %arg43, %204 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%294) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%295) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %286, %293, %277, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:307:14: warning: Rewrite for-op failed. Could not find PtrState for iter-arg index 8
    %177:5 = scf.for %arg34 = %c0_i32 to %160 step %c64_i32 iter_args(%arg35 = %150#0, %arg36 = %150#1, %arg37 = %150#2, %arg38 = %159, %arg39 = %157) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)  : i32 {
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:307:14: note: see current operation: 
%205:13 = scf.for %arg34 = %c0_i32 to %188 step %c64_i32 iter_args(%arg35 = %176#0, %arg36 = %176#1, %arg37 = %176#2, %arg38 = %structured_19, %arg39 = %offsets_20#0, %arg40 = %offsets_20#1, %arg41 = %strides_21#0, %arg42 = %strides_21#1, %arg43 = %structured_22, %arg44 = %offsets_23#0, %arg45 = %offsets_23#1, %arg46 = %strides_24#0, %arg47 = %strides_24#1) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index)  : i32 {
  %246 = arith.addi %181, %arg34 : i32
  %247 = tt.splat %246 : i32 -> tensor<64xi32>
  %248 = arith.addi %247, %66 : tensor<64xi32>
  %249 = arith.cmpi slt, %248, %189 : tensor<64xi32>
  %250 = tt.expand_dims %249 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  %251 = tt.broadcast %250 : tensor<1x64xi1> -> tensor<16x64xi1>
  %252 = tt.load %arg43, %251 : tensor<16x64x!tt.ptr<f32>>
  %253 = tt.expand_dims %248 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %254 = arith.addi %253, %cst_1 : tensor<1x64xi32>
  %255 = tt.broadcast %254 : tensor<1x64xi32> -> tensor<128x64xi32>
  %256 = arith.subi %255, %191 : tensor<128x64xi32>
  %257 = arith.subi %256, %cst_2 : tensor<128x64xi32>
  %258 = arith.addi %257, %195 : tensor<128x64xi32>
  %259 = arith.addi %258, %199 : tensor<128x64xi32>
  %260 = tt.bitcast %257 : tensor<128x64xi32> -> tensor<128x64xi32>
  %261 = arith.cmpi ult, %260, %cst_2 : tensor<128x64xi32>
  %262 = tt.addptr %200, %259 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %263 = tt.load %262, %261 : tensor<128x64x!tt.ptr<f32>>
  %264 = arith.select %261, %263, %cst_3 : tensor<128x64xi1>, tensor<128x64xf32>
  %265 = tt.dot %112, %252, %264, inputPrecision = tf32 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
  %266 = tt.broadcast %253 : tensor<1x64xi32> -> tensor<128x64xi32>
  %267 = arith.subi %191, %266 : tensor<128x64xi32>
  %268 = arith.cmpi sge, %267, %cst_4 : tensor<128x64xi32>
  %269 = arith.cmpi slt, %267, %cst_2 : tensor<128x64xi32>
  %270 = arith.andi %268, %269 : tensor<128x64xi1>
  %271 = tt.broadcast %250 : tensor<1x64xi1> -> tensor<128x64xi1>
  %272 = arith.andi %270, %271 : tensor<128x64xi1>
  %273 = arith.mulf %265, %201 : tensor<128x64xf32>
  %274 = arith.select %272, %cst_3, %cst_5 : tensor<128x64xi1>, tensor<128x64xf32>
  %275 = arith.addf %273, %274 : tensor<128x64xf32>
  %276 = "tt.reduce"(%275) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %296 = arith.maxnumf %arg48, %arg49 : f32
    tt.reduce.return %296 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %277 = arith.maxnumf %arg37, %276 : tensor<128xf32>
  %278 = tt.expand_dims %277 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %279 = tt.broadcast %278 : tensor<128x1xf32> -> tensor<128x64xf32>
  %280 = arith.subf %275, %279 : tensor<128x64xf32>
  %281 = math.exp2 %280 : tensor<128x64xf32>
  %282 = "tt.reduce"(%281) <{axis = 1 : i32}> ({
  ^bb0(%arg48: f32, %arg49: f32):
    %296 = arith.addf %arg48, %arg49 : f32
    tt.reduce.return %296 : f32
  }) : (tensor<128x64xf32>) -> tensor<128xf32>
  %283 = arith.subf %arg37, %277 : tensor<128xf32>
  %284 = math.exp2 %283 : tensor<128xf32>
  %285 = arith.mulf %arg35, %284 : tensor<128xf32>
  %286 = arith.addf %285, %282 : tensor<128xf32>
  %287 = tt.expand_dims %284 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
  %288 = tt.broadcast %287 : tensor<128x1xf32> -> tensor<128x16xf32>
  %289 = arith.mulf %arg36, %288 : tensor<128x16xf32>
  %290 = tt.expand_dims %249 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  %291 = tt.broadcast %290 : tensor<64x1xi1> -> tensor<64x16xi1>
  %292 = tt.load %arg38, %291, %cst_0 : tensor<64x16x!tt.ptr<f32>>
  %293 = tt.dot %281, %292, %289, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x16xf32> -> tensor<128x16xf32>
  %294 = tt.addptr %arg38, %203 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
  %295 = tt.addptr %arg43, %204 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%294) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
  %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%295) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
  scf.yield %286, %293, %277, %structured_25, %offsets_26#0, %offsets_26#1, %strides_27#0, %strides_27#1, %structured_28, %offsets_29#0, %offsets_29#1, %strides_30#0, %strides_30#1 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, index, index, index, index, tensor<16x64x!tt.ptr<f32>>, index, index, index, index
}
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:314:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %224 = tt.load %arg39, %223 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:314:14: note: see current operation: %252 = tt.load %arg43, %251 : tensor<16x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:314:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %224 = tt.load %arg39, %223 : tensor<16x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:314:14: note: see current operation: %252 = tt.load %arg43, %251 : tensor<16x64x!tt.ptr<f32>>
PtrAnalysis: encountered addptr operand produced by an unsupported operation
%257 = arith.subi %256, %cst_2 : tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:324:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %234 = tt.addptr %172, %231 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:324:14: note: see current operation: %262 = tt.addptr %200, %259 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:325:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %235 = tt.load %234, %233 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:325:14: note: see current operation: %263 = tt.load %262, %261 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:325:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %235 = tt.load %234, %233 : tensor<128x64x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:325:14: note: see current operation: %263 = tt.load %262, %261 : tensor<128x64x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:362:14: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
      %264 = tt.load %arg38, %263, %cst_0 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:362:14: note: see current operation: %292 = tt.load %arg38, %291, %cst_0 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:362:14: remark: PtrAnalysis: Failed to rewrite LoadOp
      %264 = tt.load %arg38, %263, %cst_0 : tensor<64x16x!tt.ptr<f32>>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:362:14: note: see current operation: %292 = tt.load %arg38, %291, %cst_0 : tensor<64x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %266 = tt.addptr %arg38, %175 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: note: see current operation: %294 = tt.addptr %arg38, %203 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      %267 = tt.addptr %arg39, %176 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: note: see current operation: %295 = tt.addptr %arg43, %204 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %266 = tt.addptr %arg38, %175 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: note: see current operation: %structured_25, %offsets_26:2, %strides_27:2 = "tts.get_structured_state"(%294) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: remark: PtrAnalysis: Failed to populate ptr state for tensor of indices
      %267 = tt.addptr %arg39, %176 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: note: see current operation: %structured_28, %offsets_29:2, %strides_30:2 = "tts.get_structured_state"(%295) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:409:5: remark: PtrAnalysis: pointer is not replace with tts.make_tptr so storeOp cannot be rewritten
    tt.store %72, %215, %195 : tensor<128x16x!tt.ptr<f32>>
    ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:409:5: note: see current operation: tt.store %98, %265, %240 : tensor<128x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:409:5: remark: PtrAnalysis: Failed to rewrite StoreOp
    tt.store %72, %215, %195 : tensor<128x16x!tt.ptr<f32>>
    ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:409:5: note: see current operation: tt.store %98, %265, %240 : tensor<128x16x!tt.ptr<f32>>
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %106 = tt.addptr %89, %105 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: note: see current operation: %structured, %offsets:2, %strides:2 = "tts.get_structured_state"(%135) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: warning: Rewriting GetStructuredStateOp failed.
    %106 = tt.addptr %89, %105 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:136:12: note: see current operation: %structured, %offsets:2, %strides:2 = "tts.get_structured_state"(%135) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %104 = tt.addptr %81, %103 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: note: see current operation: %structured_10, %offsets_11:2, %strides_12:2 = "tts.get_structured_state"(%133) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: warning: Rewriting GetStructuredStateOp failed.
    %104 = tt.addptr %81, %103 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:134:12: note: see current operation: %structured_10, %offsets_11:2, %strides_12:2 = "tts.get_structured_state"(%133) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %259 = tt.addptr %arg38, %121 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%316) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: warning: Rewriting GetStructuredStateOp failed.
      %259 = tt.addptr %arg38, %121 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:203:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%316) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %260 = tt.addptr %arg39, %122 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%317) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: warning: Rewriting GetStructuredStateOp failed.
      %260 = tt.addptr %arg39, %122 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:204:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%317) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %132 = tt.addptr %89, %131 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: note: see current operation: %structured_13, %offsets_14:2, %strides_15:2 = "tts.get_structured_state"(%163) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: warning: Rewriting GetStructuredStateOp failed.
    %132 = tt.addptr %89, %131 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:215:12: note: see current operation: %structured_13, %offsets_14:2, %strides_15:2 = "tts.get_structured_state"(%163) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %130 = tt.addptr %81, %129 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: note: see current operation: %structured_16, %offsets_17:2, %strides_18:2 = "tts.get_structured_state"(%161) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: warning: Rewriting GetStructuredStateOp failed.
    %130 = tt.addptr %81, %129 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:213:12: note: see current operation: %structured_16, %offsets_17:2, %strides_18:2 = "tts.get_structured_state"(%161) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %253 = tt.addptr %arg38, %148 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%310) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: warning: Rewriting GetStructuredStateOp failed.
      %253 = tt.addptr %arg38, %148 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:277:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%310) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %254 = tt.addptr %arg39, %149 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%311) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: warning: Rewriting GetStructuredStateOp failed.
      %254 = tt.addptr %arg39, %149 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:278:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%311) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %159 = tt.addptr %89, %158 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: note: see current operation: %structured_19, %offsets_20:2, %strides_21:2 = "tts.get_structured_state"(%192) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: warning: Rewriting GetStructuredStateOp failed.
    %159 = tt.addptr %89, %158 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:289:12: note: see current operation: %structured_19, %offsets_20:2, %strides_21:2 = "tts.get_structured_state"(%192) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
    %157 = tt.addptr %81, %156 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: note: see current operation: %structured_22, %offsets_23:2, %strides_24:2 = "tts.get_structured_state"(%190) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: warning: Rewriting GetStructuredStateOp failed.
    %157 = tt.addptr %81, %156 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
           ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:287:12: note: see current operation: %structured_22, %offsets_23:2, %strides_24:2 = "tts.get_structured_state"(%190) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %266 = tt.addptr %arg38, %175 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%323) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: warning: Rewriting GetStructuredStateOp failed.
      %266 = tt.addptr %arg38, %175 : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:364:14: note: see current operation: %structured_27, %offsets_28:2, %strides_29:2 = "tts.get_structured_state"(%323) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<64x16x!tt.ptr<f32>>) -> (tensor<64x16x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: remark: Rewrite GetStructuredStateOp failed. Could not find PtrState.
      %267 = tt.addptr %arg39, %176 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%324) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: warning: Rewriting GetStructuredStateOp failed.
      %267 = tt.addptr %arg39, %176 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
             ^
/home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir:365:14: note: see current operation: %structured_30, %offsets_31:2, %strides_32:2 = "tts.get_structured_state"(%324) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<16x64x!tt.ptr<f32>>) -> (tensor<16x64x!tt.ptr<f32>>, index, index, index, index)
"builtin.module"() ({
  "tt.func"() <{function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, f32, !tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !tt.ptr<f32>, i32, i32, i32) -> (), sym_name = "_attention_forward", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<f32>, %arg7: !tt.ptr<f32>, %arg8: !tt.ptr<i32>, %arg9: !tt.ptr<i32>, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: !tt.ptr<i32>, %arg13: !tt.ptr<i32>, %arg14: !tt.ptr<i32>, %arg15: !tt.ptr<i32>, %arg16: !tt.ptr<i32>, %arg17: !tt.ptr<i32>, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: !tt.ptr<f32>, %arg31: i32, %arg32: i32, %arg33: i32):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %16 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 128 : index}> : () -> index
    %19 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %21 = "arith.constant"() <{value = 128 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = dense<2> : tensor<128xi32>}> : () -> tensor<128xi32>
    %24 = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %25 = "arith.constant"() <{value = 1.44269502 : f32}> : () -> f32
    %26 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %28 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %29 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %30 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<128x16xf32>}> : () -> tensor<128x16xf32>
    %31 = "arith.constant"() <{value = dense<-1.000000e+06> : tensor<128x64xf32>}> : () -> tensor<128x64xf32>
    %32 = "arith.constant"() <{value = dense<0> : tensor<128x64xi32>}> : () -> tensor<128x64xi32>
    %33 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<128x64xf32>}> : () -> tensor<128x64xf32>
    %34 = "arith.constant"() <{value = dense<1> : tensor<128x64xi32>}> : () -> tensor<128x64xi32>
    %35 = "arith.constant"() <{value = dense<1> : tensor<1x64xi32>}> : () -> tensor<1x64xi32>
    %36 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %37 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x16xf32>}> : () -> tensor<64x16xf32>
    %38 = "arith.constant"() <{value = dense<0xFF800000> : tensor<128xf32>}> : () -> tensor<128xf32>
    %39 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %40 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %42 = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32
    %43 = "arith.extsi"(%42) : (i32) -> i64
    %44 = "tt.get_program_id"() <{axis = 1 : i32}> : () -> i32
    %45 = "tt.addptr"(%10, %44) : (i32, i32) -> !tt.ptr<i32>
    %46 = "tt.load"(%45) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %47 = "arith.shrsi"(%46, %20) : (i32, i32) -> i32
    %48 = "arith.cmpi"(%46, %19) <{predicate = 0 : i64}> : (i32, i32) -> i1
    "cf.cond_br"(%48)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "tt.return"() : () -> ()
  ^bb2:  // pred: ^bb0
    %49 = "tt.addptr"(%7, %46) : (i32, i32) -> !tt.ptr<i32>
    %50 = "tt.load"(%49) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %51 = "arith.extsi"(%50) : (i32) -> i64
    %52 = "arith.extsi"(%arg20) : (i32) -> i64
    %53 = "arith.muli"(%51, %52) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %54 = "arith.extsi"(%arg18) : (i32) -> i64
    %55 = "arith.muli"(%43, %54) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %56 = "arith.addi"(%53, %55) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %57 = "arith.index_cast"(%56) : (i64) -> index
    %58 = "tt.addptr"(%15, %56) : (i32, i64) -> !tt.ptr<f32>
    %59 = "tt.addptr"(%17, %56) : (i32, i64) -> !tt.ptr<f32>
    %60 = "arith.extsi"(%arg28) : (i32) -> i64
    %61 = "arith.muli"(%43, %60) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %62 = "arith.addi"(%51, %61) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %63 = "arith.index_cast"(%62) : (i64) -> index
    %64 = "arith.extsi"(%arg22) : (i32) -> i64
    %65 = "arith.muli"(%43, %64) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %66 = "arith.muli"(%47, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %67 = "arith.index_cast"(%66) : (i32) -> index
    %68 = "tt.make_range"() <{end = 128 : i32, start = 0 : i32}> : () -> tensor<128xi32>
    %69 = "tt.splat"(%66) : (i32) -> tensor<128xi32>
    %70 = "arith.addi"(%69, %68) <{overflowFlags = #arith.overflow<none>}> : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %71 = "tt.addptr"(%9, %46) : (i32, i32) -> !tt.ptr<i32>
    %72 = "tt.load"(%71) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %73 = "arith.muli"(%72, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %74 = "tt.splat"(%73) : (i32) -> tensor<128xi32>
    %75 = "arith.cmpi"(%70, %74) <{predicate = 2 : i64}> : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %76 = "arith.select"(%75, %70, %69) : (tensor<128xi1>, tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %77 = "arith.divsi"(%76, %23) : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %78 = "arith.remsi"(%76, %23) : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %79 = "tt.addptr"(%8, %46) : (i32, i32) -> !tt.ptr<i32>
    %80 = "tt.load"(%79) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %81 = "tt.addptr"(%5, %46) : (i32, i32) -> !tt.ptr<i32>
    %82 = "tt.load"(%81) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %83 = "arith.extsi"(%82) : (i32) -> i64
    %84 = "arith.extsi"(%arg23) : (i32) -> i64
    %85 = "arith.muli"(%83, %84) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %86 = "arith.addi"(%65, %85) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %87 = "tt.addptr"(%14, %86) : (i32, i64) -> !tt.ptr<f32>
    %88 = "tt.addptr"(%13, %86) : (i32, i64) -> !tt.ptr<f32>
    %89 = "tt.addptr"(%3, %46) : (i32, i32) -> !tt.ptr<i32>
    %90 = "tt.load"(%89) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %91 = "arith.index_cast"(%90) : (i32) -> index
    %92 = "tt.make_range"() <{end = 64 : i32, start = 0 : i32}> : () -> tensor<64xi32>
    %93 = "tts.make_tptr"(%arg16, %91) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, order = array<i32>, sizes = array<i64: 64>, static_offsets = array<i64: -9223372036854775808>, static_shape = array<i64: 0>, static_strides = array<i64: 1>}> : (!tt.ptr<i32>, index) -> tensor<64x!tt.ptr<i32>>
    %94 = "tts.load"(%93) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
    %95 = "tt.make_range"() <{end = 16 : i32, start = 0 : i32}> : () -> tensor<16xi32>
    %96 = "arith.divsi"(%51, %24) : (i64, i64) -> i64
    %97 = "arith.extsi"(%arg33) : (i32) -> i64
    %98 = "arith.muli"(%96, %97) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %99 = "arith.extsi"(%arg31) : (i32) -> i64
    %100 = "arith.muli"(%43, %99) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %101 = "arith.addi"(%98, %100) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %102 = "tt.addptr"(%0, %101) : (i32, i64) -> !tt.ptr<f32>
    %103 = "tt.expand_dims"(%76) <{axis = 1 : i32}> : (tensor<128xi32>) -> tensor<128x1xi32>
    %104 = "tt.splat"(%arg20) : (i32) -> tensor<128x1xi32>
    %105 = "arith.muli"(%103, %104) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
    %106 = "tt.splat"(%58) : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
    %107 = "tt.addptr"(%106, %105) : (tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>) -> tensor<128x1x!tt.ptr<f32>>
    %108 = "tt.expand_dims"(%95) <{axis = 0 : i32}> : (tensor<16xi32>) -> tensor<1x16xi32>
    %109 = "tt.broadcast"(%107) : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x16x!tt.ptr<f32>>
    %110 = "tt.broadcast"(%108) : (tensor<1x16xi32>) -> tensor<128x16xi32>
    %111 = "tt.addptr"(%109, %110) : (tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>) -> tensor<128x16x!tt.ptr<f32>>
    %112 = "tt.splat"(%59) : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
    %113 = "tt.addptr"(%112, %105) : (tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>) -> tensor<128x1x!tt.ptr<f32>>
    %114 = "tt.broadcast"(%113) : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x16x!tt.ptr<f32>>
    %115 = "tt.addptr"(%114, %110) : (tensor<128x16x!tt.ptr<f32>>, tensor<128x16xi32>) -> tensor<128x16x!tt.ptr<f32>>
    %116 = "tt.expand_dims"(%94) <{axis = 0 : i32}> : (tensor<64xi32>) -> tensor<1x64xi32>
    %117 = "tt.splat"(%arg23) : (i32) -> tensor<1x64xi32>
    %118 = "arith.muli"(%116, %117) <{overflowFlags = #arith.overflow<none>}> : (tensor<1x64xi32>, tensor<1x64xi32>) -> tensor<1x64xi32>
    %119 = "tt.splat"(%87) : (!tt.ptr<f32>) -> tensor<1x64x!tt.ptr<f32>>
    %120 = "tt.addptr"(%119, %118) : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>) -> tensor<1x64x!tt.ptr<f32>>
    %121 = "tt.expand_dims"(%95) <{axis = 1 : i32}> : (tensor<16xi32>) -> tensor<16x1xi32>
    %122 = "tt.broadcast"(%120) : (tensor<1x64x!tt.ptr<f32>>) -> tensor<16x64x!tt.ptr<f32>>
    %123 = "tt.broadcast"(%121) : (tensor<16x1xi32>) -> tensor<16x64xi32>
    %124 = "tt.addptr"(%122, %123) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
    %125 = "tt.expand_dims"(%94) <{axis = 1 : i32}> : (tensor<64xi32>) -> tensor<64x1xi32>
    %126 = "tt.splat"(%arg23) : (i32) -> tensor<64x1xi32>
    %127 = "arith.muli"(%125, %126) <{overflowFlags = #arith.overflow<none>}> : (tensor<64x1xi32>, tensor<64x1xi32>) -> tensor<64x1xi32>
    %128 = "tt.splat"(%88) : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>>
    %129 = "tt.addptr"(%128, %127) : (tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>) -> tensor<64x1x!tt.ptr<f32>>
    %130 = "tt.broadcast"(%129) : (tensor<64x1x!tt.ptr<f32>>) -> tensor<64x16x!tt.ptr<f32>>
    %131 = "tt.broadcast"(%108) : (tensor<1x16xi32>) -> tensor<64x16xi32>
    %132 = "tt.addptr"(%130, %131) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
    %133 = "arith.mulf"(%arg5, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %134 = "tt.load"(%111) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<128x16x!tt.ptr<f32>>) -> tensor<128x16xf32>
    %135 = "tt.addptr"(%4, %96) : (i32, i64) -> !tt.ptr<i32>
    %136 = "tt.splat"(%135) : (!tt.ptr<i32>) -> tensor<128x!tt.ptr<i32>>
    %137 = "tt.addptr"(%136, %77) : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>) -> tensor<128x!tt.ptr<i32>>
    %138 = "tt.load"(%137) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<128x!tt.ptr<i32>>) -> tensor<128xi32>
    %139 = "arith.muli"(%44, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %140 = "tt.addptr"(%1, %139) : (i32, i32) -> !tt.ptr<i32>
    %141 = "tt.addptr"(%140, %27) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %142 = "tt.addptr"(%140, %39) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %143 = "tt.load"(%141) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %144 = "tt.load"(%142) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %145 = "arith.muli"(%143, %arg23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %146 = "tt.splat"(%145) : (i32) -> tensor<16x64xi32>
    %147 = "tt.addptr"(%124, %146) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
    %148 = "tt.splat"(%145) : (i32) -> tensor<64x16xi32>
    %149 = "tt.addptr"(%132, %148) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
    %150 = "arith.subi"(%144, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %151 = "tt.expand_dims"(%138) <{axis = 1 : i32}> : (tensor<128xi32>) -> tensor<128x1xi32>
    %152 = "tt.broadcast"(%151) : (tensor<128x1xi32>) -> tensor<128x64xi32>
    %153 = "tt.expand_dims"(%78) <{axis = 1 : i32}> : (tensor<128xi32>) -> tensor<128x1xi32>
    %154 = "tt.splat"(%arg32) : (i32) -> tensor<128x1xi32>
    %155 = "arith.muli"(%153, %154) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
    %156 = "tt.broadcast"(%155) : (tensor<128x1xi32>) -> tensor<128x64xi32>
    %157 = "tt.expand_dims"(%77) <{axis = 1 : i32}> : (tensor<128xi32>) -> tensor<128x1xi32>
    %158 = "tt.splat"(%arg33) : (i32) -> tensor<128x1xi32>
    %159 = "arith.muli"(%157, %158) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
    %160 = "tt.broadcast"(%159) : (tensor<128x1xi32>) -> tensor<128x64xi32>
    %161 = "tt.splat"(%102) : (!tt.ptr<f32>) -> tensor<128x64x!tt.ptr<f32>>
    %162 = "tt.splat"(%133) : (f32) -> tensor<128x64xf32>
    %163 = "arith.muli"(%arg23, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %164 = "tt.splat"(%163) : (i32) -> tensor<64x16xi32>
    %165 = "tt.splat"(%163) : (i32) -> tensor<16x64xi32>
    %166:5 = "scf.for"(%27, %150, %36, %29, %30, %38, %149, %147) ({
    ^bb0(%arg54: i32, %arg55: tensor<128xf32>, %arg56: tensor<128x16xf32>, %arg57: tensor<128xf32>, %arg58: tensor<64x16x!tt.ptr<f32>>, %arg59: tensor<16x64x!tt.ptr<f32>>):
      %321 = "arith.addi"(%143, %arg54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %322 = "tt.splat"(%321) : (i32) -> tensor<64xi32>
      %323 = "arith.addi"(%322, %92) <{overflowFlags = #arith.overflow<none>}> : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi32>
      %324 = "tt.load"(%arg59) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<16x64x!tt.ptr<f32>>) -> tensor<16x64xf32>
      %325 = "tt.expand_dims"(%323) <{axis = 0 : i32}> : (tensor<64xi32>) -> tensor<1x64xi32>
      %326 = "arith.addi"(%325, %35) <{overflowFlags = #arith.overflow<none>}> : (tensor<1x64xi32>, tensor<1x64xi32>) -> tensor<1x64xi32>
      %327 = "tt.broadcast"(%326) : (tensor<1x64xi32>) -> tensor<128x64xi32>
      %328 = "arith.subi"(%327, %152) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %329 = "arith.subi"(%328, %34) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %330 = "arith.addi"(%329, %156) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %331 = "arith.addi"(%330, %160) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %332 = "tt.bitcast"(%329) : (tensor<128x64xi32>) -> tensor<128x64xi32>
      %333 = "arith.cmpi"(%332, %34) <{predicate = 6 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %334 = "tt.addptr"(%161, %331) : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>) -> tensor<128x64x!tt.ptr<f32>>
      %335 = "tt.load"(%334, %333) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi1>) -> tensor<128x64xf32>
      %336 = "arith.select"(%333, %335, %33) : (tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %337 = "tt.dot"(%134, %324, %336) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x16xf32>, tensor<16x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %338 = "tt.broadcast"(%325) : (tensor<1x64xi32>) -> tensor<128x64xi32>
      %339 = "arith.subi"(%152, %338) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %340 = "arith.cmpi"(%339, %32) <{predicate = 5 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %341 = "arith.cmpi"(%339, %34) <{predicate = 2 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %342 = "arith.andi"(%340, %341) : (tensor<128x64xi1>, tensor<128x64xi1>) -> tensor<128x64xi1>
      %343 = "arith.mulf"(%337, %162) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %344 = "arith.select"(%342, %33, %31) : (tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %345 = "arith.addf"(%343, %344) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %346 = "tt.reduce"(%345) <{axis = 1 : i32}> ({
      ^bb0(%arg62: f32, %arg63: f32):
        %365 = "arith.maxnumf"(%arg62, %arg63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%365) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %347 = "arith.maxnumf"(%arg57, %346) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %348 = "tt.expand_dims"(%347) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %349 = "tt.broadcast"(%348) : (tensor<128x1xf32>) -> tensor<128x64xf32>
      %350 = "arith.subf"(%345, %349) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %351 = "math.exp2"(%350) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>) -> tensor<128x64xf32>
      %352 = "tt.reduce"(%351) <{axis = 1 : i32}> ({
      ^bb0(%arg60: f32, %arg61: f32):
        %364 = "arith.addf"(%arg60, %arg61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%364) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %353 = "arith.subf"(%arg57, %347) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %354 = "math.exp2"(%353) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
      %355 = "arith.mulf"(%arg55, %354) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %356 = "arith.addf"(%355, %352) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %357 = "tt.expand_dims"(%354) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %358 = "tt.broadcast"(%357) : (tensor<128x1xf32>) -> tensor<128x16xf32>
      %359 = "arith.mulf"(%arg56, %358) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %360 = "tt.load"(%arg58) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<64x16x!tt.ptr<f32>>) -> tensor<64x16xf32>
      %361 = "tt.dot"(%351, %360, %359) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x64xf32>, tensor<64x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %362 = "tt.addptr"(%arg58, %164) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
      %363 = "tt.addptr"(%arg59, %165) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
      "scf.yield"(%356, %361, %347, %362, %363) : (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> ()
    }) : (i32, i32, i32, tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)
    %167 = "tt.addptr"(%140, %22) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %168 = "tt.addptr"(%140, %40) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %169 = "tt.load"(%167) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %170 = "tt.load"(%168) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %171 = "arith.muli"(%169, %arg23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %172 = "tt.splat"(%171) : (i32) -> tensor<16x64xi32>
    %173 = "tt.addptr"(%124, %172) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
    %174 = "tt.splat"(%171) : (i32) -> tensor<64x16xi32>
    %175 = "tt.addptr"(%132, %174) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
    %176 = "arith.subi"(%170, %169) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %177 = "tt.splat"(%133) : (f32) -> tensor<128xf32>
    %178:5 = "scf.for"(%27, %176, %36, %166#0, %166#1, %166#2, %175, %173) ({
    ^bb0(%arg44: i32, %arg45: tensor<128xf32>, %arg46: tensor<128x16xf32>, %arg47: tensor<128xf32>, %arg48: tensor<64x16x!tt.ptr<f32>>, %arg49: tensor<16x64x!tt.ptr<f32>>):
      %282 = "arith.addi"(%169, %arg44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %283 = "tt.splat"(%282) : (i32) -> tensor<64xi32>
      %284 = "arith.addi"(%283, %92) <{overflowFlags = #arith.overflow<none>}> : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi32>
      %285 = "tt.load"(%arg49) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<16x64x!tt.ptr<f32>>) -> tensor<16x64xf32>
      %286 = "tt.expand_dims"(%284) <{axis = 0 : i32}> : (tensor<64xi32>) -> tensor<1x64xi32>
      %287 = "arith.addi"(%286, %35) <{overflowFlags = #arith.overflow<none>}> : (tensor<1x64xi32>, tensor<1x64xi32>) -> tensor<1x64xi32>
      %288 = "tt.broadcast"(%287) : (tensor<1x64xi32>) -> tensor<128x64xi32>
      %289 = "arith.subi"(%288, %152) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %290 = "arith.subi"(%289, %34) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %291 = "arith.addi"(%290, %156) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %292 = "arith.addi"(%291, %160) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %293 = "tt.bitcast"(%290) : (tensor<128x64xi32>) -> tensor<128x64xi32>
      %294 = "arith.cmpi"(%293, %34) <{predicate = 6 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %295 = "tt.addptr"(%161, %292) : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>) -> tensor<128x64x!tt.ptr<f32>>
      %296 = "tt.load"(%295, %294) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi1>) -> tensor<128x64xf32>
      %297 = "arith.select"(%294, %296, %33) : (tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %298 = "tt.dot"(%134, %285, %297) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x16xf32>, tensor<16x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %299 = "tt.reduce"(%298) <{axis = 1 : i32}> ({
      ^bb0(%arg52: f32, %arg53: f32):
        %320 = "arith.maxnumf"(%arg52, %arg53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%320) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %300 = "arith.mulf"(%299, %177) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %301 = "arith.maxnumf"(%arg47, %300) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %302 = "arith.mulf"(%298, %162) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %303 = "tt.expand_dims"(%301) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %304 = "tt.broadcast"(%303) : (tensor<128x1xf32>) -> tensor<128x64xf32>
      %305 = "arith.subf"(%302, %304) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %306 = "math.exp2"(%305) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>) -> tensor<128x64xf32>
      %307 = "tt.reduce"(%306) <{axis = 1 : i32}> ({
      ^bb0(%arg50: f32, %arg51: f32):
        %319 = "arith.addf"(%arg50, %arg51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%319) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %308 = "arith.subf"(%arg47, %301) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %309 = "math.exp2"(%308) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
      %310 = "arith.mulf"(%arg45, %309) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %311 = "arith.addf"(%310, %307) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %312 = "tt.expand_dims"(%309) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %313 = "tt.broadcast"(%312) : (tensor<128x1xf32>) -> tensor<128x16xf32>
      %314 = "arith.mulf"(%arg46, %313) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %315 = "tt.load"(%arg48) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<64x16x!tt.ptr<f32>>) -> tensor<64x16xf32>
      %316 = "tt.dot"(%306, %315, %314) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x64xf32>, tensor<64x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %317 = "tt.addptr"(%arg48, %164) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
      %318 = "tt.addptr"(%arg49, %165) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
      "scf.yield"(%311, %316, %301, %317, %318) : (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> ()
    }) : (i32, i32, i32, tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)
    %179 = "tt.addptr"(%140, %28) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %180 = "tt.addptr"(%140, %41) : (!tt.ptr<i32>, i32) -> !tt.ptr<i32>
    %181 = "tt.load"(%179) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %182 = "tt.load"(%180) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<i32>) -> i32
    %183 = "arith.muli"(%181, %arg23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %184 = "tt.splat"(%183) : (i32) -> tensor<16x64xi32>
    %185 = "tt.addptr"(%124, %184) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
    %186 = "tt.splat"(%183) : (i32) -> tensor<64x16xi32>
    %187 = "tt.addptr"(%132, %186) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
    %188 = "arith.subi"(%182, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %189 = "tt.splat"(%80) : (i32) -> tensor<64xi32>
    %190:5 = "scf.for"(%27, %188, %36, %178#0, %178#1, %178#2, %187, %185) ({
    ^bb0(%arg34: i32, %arg35: tensor<128xf32>, %arg36: tensor<128x16xf32>, %arg37: tensor<128xf32>, %arg38: tensor<64x16x!tt.ptr<f32>>, %arg39: tensor<16x64x!tt.ptr<f32>>):
      %230 = "arith.addi"(%181, %arg34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %231 = "tt.splat"(%230) : (i32) -> tensor<64xi32>
      %232 = "arith.addi"(%231, %92) <{overflowFlags = #arith.overflow<none>}> : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi32>
      %233 = "arith.cmpi"(%232, %189) <{predicate = 2 : i64}> : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi1>
      %234 = "tt.expand_dims"(%233) <{axis = 0 : i32}> : (tensor<64xi1>) -> tensor<1x64xi1>
      %235 = "tt.broadcast"(%234) : (tensor<1x64xi1>) -> tensor<16x64xi1>
      %236 = "tt.load"(%arg39, %235) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi1>) -> tensor<16x64xf32>
      %237 = "tt.expand_dims"(%232) <{axis = 0 : i32}> : (tensor<64xi32>) -> tensor<1x64xi32>
      %238 = "arith.addi"(%237, %35) <{overflowFlags = #arith.overflow<none>}> : (tensor<1x64xi32>, tensor<1x64xi32>) -> tensor<1x64xi32>
      %239 = "tt.broadcast"(%238) : (tensor<1x64xi32>) -> tensor<128x64xi32>
      %240 = "arith.subi"(%239, %152) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %241 = "arith.subi"(%240, %34) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %242 = "arith.addi"(%241, %156) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %243 = "arith.addi"(%242, %160) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %244 = "tt.bitcast"(%241) : (tensor<128x64xi32>) -> tensor<128x64xi32>
      %245 = "arith.cmpi"(%244, %34) <{predicate = 6 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %246 = "tt.addptr"(%161, %243) : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>) -> tensor<128x64x!tt.ptr<f32>>
      %247 = "tt.load"(%246, %245) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi1>) -> tensor<128x64xf32>
      %248 = "arith.select"(%245, %247, %33) : (tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %249 = "tt.dot"(%134, %236, %248) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x16xf32>, tensor<16x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %250 = "tt.broadcast"(%237) : (tensor<1x64xi32>) -> tensor<128x64xi32>
      %251 = "arith.subi"(%152, %250) <{overflowFlags = #arith.overflow<none>}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi32>
      %252 = "arith.cmpi"(%251, %32) <{predicate = 5 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %253 = "arith.cmpi"(%251, %34) <{predicate = 2 : i64}> : (tensor<128x64xi32>, tensor<128x64xi32>) -> tensor<128x64xi1>
      %254 = "arith.andi"(%252, %253) : (tensor<128x64xi1>, tensor<128x64xi1>) -> tensor<128x64xi1>
      %255 = "tt.broadcast"(%234) : (tensor<1x64xi1>) -> tensor<128x64xi1>
      %256 = "arith.andi"(%254, %255) : (tensor<128x64xi1>, tensor<128x64xi1>) -> tensor<128x64xi1>
      %257 = "arith.mulf"(%249, %162) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %258 = "arith.select"(%256, %33, %31) : (tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %259 = "arith.addf"(%257, %258) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %260 = "tt.reduce"(%259) <{axis = 1 : i32}> ({
      ^bb0(%arg42: f32, %arg43: f32):
        %281 = "arith.maxnumf"(%arg42, %arg43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%281) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %261 = "arith.maxnumf"(%arg37, %260) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %262 = "tt.expand_dims"(%261) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %263 = "tt.broadcast"(%262) : (tensor<128x1xf32>) -> tensor<128x64xf32>
      %264 = "arith.subf"(%259, %263) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
      %265 = "math.exp2"(%264) <{fastmath = #arith.fastmath<none>}> : (tensor<128x64xf32>) -> tensor<128x64xf32>
      %266 = "tt.reduce"(%265) <{axis = 1 : i32}> ({
      ^bb0(%arg40: f32, %arg41: f32):
        %280 = "arith.addf"(%arg40, %arg41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "tt.reduce.return"(%280) : (f32) -> ()
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %267 = "arith.subf"(%arg37, %261) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %268 = "math.exp2"(%267) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
      %269 = "arith.mulf"(%arg35, %268) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %270 = "arith.addf"(%269, %266) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %271 = "tt.expand_dims"(%268) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
      %272 = "tt.broadcast"(%271) : (tensor<128x1xf32>) -> tensor<128x16xf32>
      %273 = "arith.mulf"(%arg36, %272) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %274 = "tt.expand_dims"(%233) <{axis = 1 : i32}> : (tensor<64xi1>) -> tensor<64x1xi1>
      %275 = "tt.broadcast"(%274) : (tensor<64x1xi1>) -> tensor<64x16xi1>
      %276 = "tt.load"(%arg38, %275, %37) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi1>, tensor<64x16xf32>) -> tensor<64x16xf32>
      %277 = "tt.dot"(%265, %276, %273) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<128x64xf32>, tensor<64x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
      %278 = "tt.addptr"(%arg38, %164) : (tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>) -> tensor<64x16x!tt.ptr<f32>>
      %279 = "tt.addptr"(%arg39, %165) : (tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>) -> tensor<16x64x!tt.ptr<f32>>
      "scf.yield"(%270, %277, %261, %278, %279) : (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> ()
    }) : (i32, i32, i32, tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16x!tt.ptr<f32>>, tensor<16x64x!tt.ptr<f32>>)
    %191 = "math.log2"(%190#0) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %192 = "arith.addf"(%190#2, %191) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %193 = "tt.expand_dims"(%190#0) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
    %194 = "tt.broadcast"(%193) : (tensor<128x1xf32>) -> tensor<128x16xf32>
    %195 = "arith.divf"(%190#1, %194) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
    %196 = "arith.index_cast"(%arg20) : (i32) -> index
    %197 = "arith.muli"(%67, %196) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %198 = "arith.addi"(%57, %197) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %199 = "tts.make_tptr"(%arg6, %196, %198) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, order = array<i32>, sizes = array<i64: 128, 16>, static_offsets = array<i64: -9223372036854775808, 0>, static_shape = array<i64: 0, 0>, static_strides = array<i64: -9223372036854775808, 1>}> : (!tt.ptr<f32>, index, index) -> tensor<128x16x!tt.ptr<f32>>
    %200 = "arith.addi"(%63, %67) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %201 = "tts.make_tptr"(%arg7, %200) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, order = array<i32>, sizes = array<i64: 128>, static_offsets = array<i64: -9223372036854775808>, static_shape = array<i64: 0>, static_strides = array<i64: 1>}> : (!tt.ptr<f32>, index) -> tensor<128x!tt.ptr<f32>>
    %202 = "arith.addi"(%67, %18) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %203 = "arith.index_cast"(%73) : (i32) -> index
    %204 = "arith.minsi"(%202, %203) : (index, index) -> index
    %205 = "arith.subi"(%204, %67) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %206 = "tts.load"(%201, %205) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, index) -> tensor<128xf32>
    %207 = "tt.expand_dims"(%75) <{axis = 1 : i32}> : (tensor<128xi1>) -> tensor<128x1xi1>
    %208 = "tt.broadcast"(%207) : (tensor<128x1xi1>) -> tensor<128x16xi1>
    %209 = "tts.load"(%199, %205) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 16>}> : (tensor<128x16x!tt.ptr<f32>>, index) -> tensor<128x16xf32>
    %210 = "arith.maxnumf"(%206, %192) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %211 = "arith.subf"(%192, %210) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %212 = "math.exp2"(%211) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %213 = "arith.subf"(%206, %210) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %214 = "math.exp2"(%213) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %215 = "arith.addf"(%212, %214) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %216 = "math.log2"(%215) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %217 = "arith.addf"(%210, %216) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %218 = "arith.subf"(%192, %217) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %219 = "math.exp2"(%218) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %220 = "tt.expand_dims"(%219) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
    %221 = "tt.broadcast"(%220) : (tensor<128x1xf32>) -> tensor<128x16xf32>
    %222 = "arith.mulf"(%195, %221) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
    %223 = "arith.subf"(%206, %217) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %224 = "math.exp2"(%223) <{fastmath = #arith.fastmath<none>}> : (tensor<128xf32>) -> tensor<128xf32>
    %225 = "tt.expand_dims"(%224) <{axis = 1 : i32}> : (tensor<128xf32>) -> tensor<128x1xf32>
    %226 = "tt.broadcast"(%225) : (tensor<128x1xf32>) -> tensor<128x16xf32>
    %227 = "arith.mulf"(%209, %226) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
    %228 = "arith.addf"(%222, %227) <{fastmath = #arith.fastmath<none>}> : (tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
    %229 = "tts.make_tptr"(%arg1, %200) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, order = array<i32>, sizes = array<i64: 128>, static_offsets = array<i64: -9223372036854775808>, static_shape = array<i64: 0>, static_strides = array<i64: 1>}> : (!tt.ptr<f32>, index) -> tensor<128x!tt.ptr<f32>>
    "tts.store"(%229, %217, %205) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>, index) -> ()
    "tt.store"(%115, %228, %208) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<128x16x!tt.ptr<f32>>, tensor<128x16xf32>, tensor<128x16xi1>) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()
processing val
%17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
these are the uses:
%59 = "tt.addptr"(%17, %56) : (i32, i64) -> !tt.ptr<f32>
actual processing
processing user
%59 = "tt.addptr"(%17, %56) : (i32, i64) -> !tt.ptr<f32>
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.	Program arguments: /home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt --triton-to-linalg-experimental /home/nhat/export-model-runner/triton-inference/___true_gather/forward_cleaned.ttir.mlir --mlir-print-ir-after-failure
 #0 0x000055c0774ebe67 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x255ee67)
 #1 0x000055c0774e998e llvm::sys::RunSignalHandlers() (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x255c98e)
 #2 0x000055c0774ec51f SignalHandler(int) Signals.cpp:0:0
 #3 0x00007fcba80d8420 __restore_rt (/lib/x86_64-linux-gnu/libpthread.so.0+0x14420)
 #4 0x000055c077408cc3 mlir::RankedTensorType::getElementType() const (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x247bcc3)
 #5 0x000055c07552f2f8 mlir::TypeID::operator==(mlir::TypeID const&) const /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/mlir/Support/TypeID.h:116:20
 #6 0x000055c07552f2f8 bool mlir::detail::StorageUserBase<mlir::RankedTensorType, mlir::TensorType, mlir::detail::RankedTensorTypeStorage, mlir::detail::TypeUniquer, mlir::ShapedType::Trait, mlir::ValueSemantics>::classof<mlir::Type>(mlir::Type) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/mlir/IR/StorageUniquerSupport.h:113:28
 #7 0x000055c07552f2f8 llvm::CastInfo<mlir::RankedTensorType, mlir::Type, void>::isPossible(mlir::Type) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/mlir/IR/Types.h:419:14
 #8 0x000055c07552f2f8 llvm::DefaultDoCastIfPossible<mlir::RankedTensorType, mlir::Type, llvm::CastInfo<mlir::RankedTensorType, mlir::Type, void>>::doCastIfPossible(mlir::Type) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/llvm/Support/Casting.h:311:10
 #9 0x000055c07552f2f8 decltype(auto) llvm::dyn_cast<mlir::RankedTensorType, mlir::Type>(mlir::Type&) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/llvm/Support/Casting.h:657:10
#10 0x000055c07552f2f8 (anonymous namespace)::getPtrOffsetType(mlir::Type, unsigned int) /home/nhat/github/triton_shared/lib/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtrPass.cpp:65:25
#11 0x000055c07552d4ae (anonymous namespace)::FoldUnstructuredTritonAddPtrPass::computePtrType(unsigned int)::'lambda'(mlir::triton::AddPtrOp)::operator()(mlir::triton::AddPtrOp) const /home/nhat/github/triton_shared/lib/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtrPass.cpp:193:26
#12 0x000055c07552d4ae llvm::TypeSwitch<mlir::Operation*, void>& llvm::TypeSwitch<mlir::Operation*, void>::Case<mlir::triton::AddPtrOp, (anonymous namespace)::FoldUnstructuredTritonAddPtrPass::computePtrType(unsigned int)::'lambda'(mlir::triton::AddPtrOp)>((anonymous namespace)::FoldUnstructuredTritonAddPtrPass::computePtrType(unsigned int)::'lambda'(mlir::triton::AddPtrOp)&&) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/llvm/ADT/TypeSwitch.h:149:7
#13 0x000055c07552d4ae (anonymous namespace)::FoldUnstructuredTritonAddPtrPass::computePtrType(unsigned int) /home/nhat/github/triton_shared/lib/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtrPass.cpp:179:14
#14 0x000055c07552d4ae (anonymous namespace)::FoldUnstructuredTritonAddPtrPass::runOnOperation() /home/nhat/github/triton_shared/lib/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtrPass.cpp:376:45
#15 0x000055c076a566b6 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac96b6)
#16 0x000055c076a56e60 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac9e60)
#17 0x000055c076a5b3af llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (mlir::OpPassManager&, mlir::Operation*)>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_6>(long, mlir::OpPassManager&, mlir::Operation*) Pass.cpp:0:0
#18 0x000055c07550ee6d llvm::LogicalResult::failed() const /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/llvm/Support/LogicalResult.h:43:43
#19 0x000055c07550ee6d llvm::failed(llvm::LogicalResult) /home/nhat/.triton/llvm/llvm-c08c6a71-ubuntu-x64/include/llvm/Support/LogicalResult.h:71:58
#20 0x000055c07550ee6d (anonymous namespace)::TritonToLinalgExperimentalPass::runOnOperation() /home/nhat/github/triton_shared/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp:67:9
#21 0x000055c076a566b6 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac96b6)
#22 0x000055c076a56e60 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac9e60)
#23 0x000055c076a59315 mlir::PassManager::run(mlir::Operation*) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1acc315)
#24 0x000055c076a529ef performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
#25 0x000055c076a5261d llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::$_2>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
#26 0x000055c07747efc6 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x24f1fc6)
#27 0x000055c076a4d441 mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac0441)
#28 0x000055c076a4d6f3 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac06f3)
#29 0x000055c076a4dac6 mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x1ac0ac6)
#30 0x000055c07555f4ab main /home/nhat/github/triton_shared/tools/triton-shared-opt/triton-shared-opt.cpp:16:33
#31 0x00007fcba7b7c083 __libc_start_main /build/glibc-LcI20x/glibc-2.31/csu/../csu/libc-start.c:342:3
#32 0x000055c07521f92e _start (/home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.8/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt+0x29292e)
