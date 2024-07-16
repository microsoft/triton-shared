tensor<4x256xbf16>
tensor<1x256x!tt.ptr<bf16>>
module {
  tt.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: i32) {
    %cst = arith.constant dense<256> : tensor<4xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x256xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16>
    %c256_i32 = arith.constant 256 : i32
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %3 = tt.addptr %2, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    %4 = tt.load %3 : tensor<1x256x!tt.ptr<bf16>>
    %5 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    %6 = tt.addptr %3, %5 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    tt.store %6, %4 : tensor<1x256x!tt.ptr<bf16>>
    %7 = builtin.unrealized_conversion_cast %3 : tensor<1x256x!tt.ptr<bf16>> to tuple<tensor<1x256x!tt.ptr<bf16>>, tuple<index, index, index, index>> {make_state}
    %8:2 = scf.for %arg2 = %c0 to %c12 step %c3 iter_args(%arg3 = %cst_1, %arg4 = %7) -> (tensor<4x256xbf16>, tuple<tensor<1x256x!tt.ptr<bf16>>, tuple<index, index, index, index>>) {
      %15 = builtin.unrealized_conversion_cast %arg4 : tuple<tensor<1x256x!tt.ptr<bf16>>, tuple<index, index, index, index>> to tensor<1x256x!tt.ptr<bf16>> {state_placeholder}
      %16 = tt.broadcast %15 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
      %17 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %18 = arith.index_cast %arg2 : index to i32
      %19 = arith.muli %18, %c256_i32 : i32
      %20 = tt.splat %19 : i32 -> tensor<4xi32>
      %21 = arith.muli %17, %20 : tensor<4xi32>
      %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %23 = tt.broadcast %22 : tensor<4x1xi32> -> tensor<4x256xi32>
      %24 = tt.addptr %16, %23 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
      %25 = tt.load %24 : tensor<4x256x!tt.ptr<bf16>>
      %26 = arith.addf %arg3, %25 : tensor<4x256xbf16>
      %27 = tt.addptr %15, %cst_0 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
      %28 = builtin.unrealized_conversion_cast %27 : tensor<1x256x!tt.ptr<bf16>> to tuple<tensor<1x256x!tt.ptr<bf16>>, tuple<index, index, index, index>> {make_state}
      scf.yield %26, %28 : tensor<4x256xbf16>, tuple<tensor<1x256x!tt.ptr<bf16>>, tuple<index, index, index, index>>
    }
    %9 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %10 = arith.muli %9, %cst : tensor<4xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %12 = tt.broadcast %11 : tensor<4x1xi32> -> tensor<4x256xi32>
    %13 = tt.broadcast %3 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %12 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    tt.store %14, %8#0 : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

