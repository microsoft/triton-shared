module {
  tt.func public @maxnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.maxnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 : !tt.ptr<f32>
    tt.return
  }
}