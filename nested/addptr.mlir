module {
  tt.func @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3:2 = "tts.state_placeholder"(%2) : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
    %4:3 = scf.for %arg5 = %c0 to %c12 step %c3 iter_args(%arg6 = %3#0, %arg7 = %3#1, %arg8 = %cst) -> (!tt.ptr<f32>, index, tensor<1024xf32>) {
      %10 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %11 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
      %14 = math.exp %13 : tensor<1024xf32>
      %15 = arith.addf %arg8, %14 : tensor<1024xf32>
      %16 = arith.index_cast %arg5 : index to i32
      %17 = tt.addptr %arg6, %16 : !tt.ptr<f32>, i32
      %18:2 = "tts.state_placeholder"(%17) : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
      scf.yield %18#0, %18#1, %15 : !tt.ptr<f32>, index, tensor<1024xf32>
    }
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %6 = arith.muli %0, %arg3 : i32
    %7 = tt.addptr %arg0, %6 : !tt.ptr<f32>, i32
    %8 = tt.splat %7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %9, %4#2 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}