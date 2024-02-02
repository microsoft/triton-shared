module {
  tt.func @kernel(%arg0: f32, %arg1: bf16, %arg2: tensor<1024x!tt.ptr<f32, 1>>, %arg3: tensor<128x256x!tt.ptr<bf16, 1>>) {
    %0 = tt.splat %arg0 : (f32) -> tensor<1024xf32>
    %1 = tt.splat %arg1 : (bf16) -> tensor<128x256xbf16>
    tt.store %arg2, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.store %arg3, %1 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
    tt.return
  }
}

