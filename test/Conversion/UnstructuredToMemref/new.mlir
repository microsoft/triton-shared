module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<64xi8>
    %cst_0 = arith.constant dense<0> : tensor<64xi8>
    %cst_1 = arith.constant dense<10> : tensor<64xi32>
    %cst_2 = arith.constant dense<5> : tensor<64xi32>
    %from_elements = tensor.from_elements %arg0, %arg1 : tensor<2x!tt.ptr<f32>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = arith.divsi %0, %cst_1 : tensor<64xi32>
    %2 = arith.addi %1, %cst_2 : tensor<64xi32>
    %3 = tts.gather %from_elements %cst_0[%2] : (tensor<2x!tt.ptr<f32>>, tensor<64xi8>, tensor<64xi32>) -> tensor<64xf32>
    tts.scatter %3 into %from_elements %cst[%0] : tensor<64xf32> into (tensor<2x!tt.ptr<f32>>, tensor<64xi8>, tensor<64xi32>)
    tt.return
  }
}