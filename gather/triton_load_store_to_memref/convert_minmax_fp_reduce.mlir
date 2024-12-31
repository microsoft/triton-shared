module {
  func.func @maxnumf(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %2 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst_0 into %2[] : tensor<f32>
    %reduced = linalg.reduce ins(%1 : tensor<4096xf32>) outs(%inserted : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %4 = arith.maxnumf %in, %init : f32
        linalg.yield %4 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %3 = "tts.make_unstructured_tptr"(%arg0, %c0_i64) : (!tt.ptr<f32>, i64) -> !tt.ptr<f32>
    tt.store %3, %extracted : !tt.ptr<f32>
    return
  }
}
