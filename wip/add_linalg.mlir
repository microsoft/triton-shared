#map = affine_map<(d0) -> (d0)>
module {
  tt.func @add_kernel_01234(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %0 : i32 to index
    %4 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %5 = arith.index_cast %0 : i32 to index
    %6 = arith.addi %5, %c1024 : index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.minsi %6, %7 : index
    %9 = arith.subi %8, %5 : index
    %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, index) -> tensor<1024xf32>
    %11 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %12 = arith.index_cast %0 : i32 to index
    %13 = arith.addi %12, %c1024 : index
    %14 = arith.index_cast %arg3 : i32 to index
    %15 = arith.minsi %13, %14 : index
    %16 = arith.subi %15, %12 : index
    %17 = "tts.load"(%11, %16) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, index) -> tensor<1024xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %17 : tensor<1024xf32>, tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %25 = arith.addf %in, %in_0 : f32
      linalg.yield %25 : f32
    } -> tensor<1024xf32>
    %19 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32, 1> to tensor<1024x!tt.ptr<f32, 1>>
    %20 = arith.index_cast %0 : i32 to index
    %21 = arith.addi %20, %c1024 : index
    %22 = arith.index_cast %arg3 : i32 to index
    %23 = arith.minsi %21, %22 : index
    %24 = arith.subi %23, %20 : index
    "tts.store"(%19, %18, %24) <{static_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xf32>, index) -> ()
    tt.return
  }
}

