module {
  tt.func public @one_to_n_loads_simplified(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32

    // offs_x_m = tl.arange(0, BLOCK_M)
    // offs_x_m = tl.load(GatherIndx + offs_x_m)
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i32>>
    // ^^^^^ We want to leave this part to PtrAnalysis ^^^^^

    // dynamic dims
    // offs_x_m[:, None] * stride_x_m
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %7 = arith.muli %5, %6 : tensor<2x1xi32>
    // Probably want to simplify this?

    // static dims
    // offs_x_k[None, :] * stride_x_k
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %10 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %11 = tt.splat %arg4 : i32 -> tensor<1x4xi32>
    %12 = arith.muli %10, %11 : tensor<1x4xi32>


    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %13 = tt.broadcast %9 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %14 = tt.broadcast %12 : tensor<1x4xi32> -> tensor<2x4xi32>

    // InPtrs = In + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
    %15 = tt.addptr %13, %14 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>


    // OutPtrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %17 = arith.muli %16, %6 : tensor<2x1xi32>
    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %17 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %20 = tt.broadcast %19 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %14 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>

    // offset
    %22 = arith.muli %arg4, %c4_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x4xi32>

    // the gather ptr needs:
    // 1) have a tensor of offsets
    // 2) structured state for the structured portion
    // actually do i need it?
    // we do, because we want to lower it to
    // %res = tensor something
    // scf %i = 0 to %n (%tensor of index) {
    //   %offset = %tensor[%i]
    //   %ptr = tts.make_tptr offsets [%offset, 0]
    //   %tensor = tts.load %ptr
    //   insert %tensor into %res
    // }
    //
    // i think the tts.get_structured_state needs to come from the static dimensions
    // so that means we need to create a tt.ptr with the static dims?
    //
    //

    %structuredPtr, %offsets:2, %strides:2 = "tts.get_structured_state"(%15) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<2x4x!tt.ptr<f32>>) -> (tensor<2x4x!tt.ptr<f32>>, index, index, index, index)
    %structuredPtr_0, %offsets_1:2, %strides_2:2 = "tts.get_structured_state"(%21) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<2x4x!tt.ptr<f32>>) -> (tensor<2x4x!tt.ptr<f32>>, index, index, index, index)
    %24:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %structuredPtr, %arg7 = %structuredPtr_0) -> (tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>)  : i32 {
      %25 = tt.load %arg6 : tensor<2x4x!tt.ptr<f32>>
      tt.store %arg7, %25 : tensor<2x4x!tt.ptr<f32>>
      %26 = tt.addptr %arg6, %23 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %27 = tt.addptr %arg7, %23 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %structuredPtr_3, %offsets_4:2, %strides_5:2 = "tts.get_structured_state"(%26) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<2x4x!tt.ptr<f32>>) -> (tensor<2x4x!tt.ptr<f32>>, index, index, index, index)
      %structuredPtr_6, %offsets_7:2, %strides_8:2 = "tts.get_structured_state"(%27) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<2x4x!tt.ptr<f32>>) -> (tensor<2x4x!tt.ptr<f32>>, index, index, index, index)
      scf.yield %structuredPtr_3, %structuredPtr_6 : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}
