#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @_attention_forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: memref<*xf32>, %arg4: memref<*xf32>, %arg5: f32, %arg6: memref<*xf32>, %arg7: memref<*xf32>, %arg8: memref<*xi32>, %arg9: memref<*xi32>, %arg10: memref<*xi32>, %arg11: memref<*xi32>, %arg12: memref<*xi32>, %arg13: memref<*xi32>, %arg14: memref<*xi32>, %arg15: memref<*xi32>, %arg16: memref<*xi32>, %arg17: memref<*xi32>, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: memref<*xf32>, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32) {
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 0xFF800000 : f32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant -1.000000e+06 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %c2_i64 = arith.constant 2 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128 = arith.constant 128 : index
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tensor.empty() : tensor<128xi32>
    %1 = linalg.fill ins(%c2_i32 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
    %2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %4 = tensor.empty() : tensor<128x16xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<128x16xf32>) -> tensor<128x16xf32>
    %6 = tensor.empty() : tensor<128x64xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %8 = tensor.empty() : tensor<128x64xi32>
    %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<128x64xi32>) -> tensor<128x64xi32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %11 = linalg.fill ins(%c1_i32 : i32) outs(%8 : tensor<128x64xi32>) -> tensor<128x64xi32>
    %12 = tensor.empty() : tensor<1x64xi32>
    %13 = linalg.fill ins(%c1_i32 : i32) outs(%12 : tensor<1x64xi32>) -> tensor<1x64xi32>
    %14 = tensor.empty() : tensor<64x16xf32>
    %15 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %16 = arith.extsi %arg37 : i32 to i64
    %17 = arith.index_cast %arg38 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%17], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = affine.load %reinterpret_cast[0] : memref<1xi32, strided<[1], offset: ?>>
    %19 = arith.shrsi %18, %c32_i32 : i32
    %20 = arith.cmpi eq, %18, %c-1_i32 : i32
    cf.cond_br %20, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    return
  ^bb2:  // pred: ^bb0
    %21 = arith.index_cast %18 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg11 to offset: [%21], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %22 = affine.load %reinterpret_cast_3[0] : memref<1xi32, strided<[1], offset: ?>>
    %23 = arith.extsi %22 : i32 to i64
    %24 = arith.extsi %arg20 : i32 to i64
    %25 = arith.muli %23, %24 : i64
    %26 = arith.extsi %arg18 : i32 to i64
    %27 = arith.muli %16, %26 : i64
    %28 = arith.addi %25, %27 : i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.extsi %arg28 : i32 to i64
    %31 = arith.muli %16, %30 : i64
    %32 = arith.addi %23, %31 : i64
    %33 = arith.index_cast %32 : i64 to index
    %34 = arith.extsi %arg22 : i32 to i64
    %35 = arith.muli %16, %34 : i64
    %36 = arith.muli %19, %c128_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<128xi32>) {
    ^bb0(%out: i32):
      %208 = linalg.index 0 : index
      %209 = arith.index_cast %208 : index to i32
      linalg.yield %209 : i32
    } -> tensor<128xi32>
    %39 = linalg.fill ins(%36 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
    %40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%39, %38 : tensor<128xi32>, tensor<128xi32>) outs(%39 : tensor<128xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.addi %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128xi32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg9 to offset: [%21], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %41 = affine.load %reinterpret_cast_4[0] : memref<1xi32, strided<[1], offset: ?>>
    %42 = arith.muli %41, %c2_i32 : i32
    %43 = linalg.fill ins(%42 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
    %44 = tensor.empty() : tensor<128xi1>
    %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%40, %43 : tensor<128xi32>, tensor<128xi32>) outs(%44 : tensor<128xi1>) {
    ^bb0(%in: i32, %in_37: i32, %out: i1):
      %208 = arith.cmpi slt, %in, %in_37 : i32
      linalg.yield %208 : i1
    } -> tensor<128xi1>
    %46 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%45, %40, %39 : tensor<128xi1>, tensor<128xi32>, tensor<128xi32>) outs(%40 : tensor<128xi32>) {
    ^bb0(%in: i1, %in_37: i32, %in_38: i32, %out: i32):
      %208 = arith.select %in, %in_37, %in_38 : i32
      linalg.yield %208 : i32
    } -> tensor<128xi32>
    %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%46, %1 : tensor<128xi32>, tensor<128xi32>) outs(%46 : tensor<128xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.divsi %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128xi32>
    %48 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%46, %1 : tensor<128xi32>, tensor<128xi32>) outs(%46 : tensor<128xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.remsi %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128xi32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg10 to offset: [%21], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %49 = affine.load %reinterpret_cast_5[0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg13 to offset: [%21], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %50 = affine.load %reinterpret_cast_6[0] : memref<1xi32, strided<[1], offset: ?>>
    %51 = arith.extsi %50 : i32 to i64
    %52 = arith.extsi %arg23 : i32 to i64
    %53 = arith.muli %51, %52 : i64
    %54 = arith.addi %35, %53 : i64
    %reinterpret_cast_7 = memref.reinterpret_cast %arg15 to offset: [%21], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %55 = affine.load %reinterpret_cast_7[0] : memref<1xi32, strided<[1], offset: ?>>
    %56 = arith.index_cast %55 : i32 to index
    %57 = tensor.empty() : tensor<64xi32>
    %58 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%57 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %208 = linalg.index 0 : index
      %209 = arith.index_cast %208 : index to i32
      linalg.yield %209 : i32
    } -> tensor<64xi32>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg16 to offset: [%56], sizes: [64], strides: [1] : memref<*xi32> to memref<64xi32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<64xi32>
    memref.copy %reinterpret_cast_8, %alloc : memref<64xi32, strided<[1], offset: ?>> to memref<64xi32>
    %59 = bufferization.to_tensor %alloc restrict writable : memref<64xi32>
    %60 = tensor.empty() : tensor<16xi32>
    %61 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%60 : tensor<16xi32>) {
    ^bb0(%out: i32):
      %208 = linalg.index 0 : index
      %209 = arith.index_cast %208 : index to i32
      linalg.yield %209 : i32
    } -> tensor<16xi32>
    %62 = arith.divsi %23, %c2_i64 : i64
    %63 = arith.extsi %arg33 : i32 to i64
    %64 = arith.muli %62, %63 : i64
    %65 = arith.extsi %arg31 : i32 to i64
    %66 = arith.muli %16, %65 : i64
    %67 = arith.addi %64, %66 : i64
    %expanded = tensor.expand_shape %46 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %68 = tensor.empty() : tensor<128x1xi32>
    %69 = linalg.fill ins(%arg20 : i32) outs(%68 : tensor<128x1xi32>) -> tensor<128x1xi32>
    %70 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded, %69 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%expanded : tensor<128x1xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.muli %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128x1xi32>
    %71 = tensor.empty() : tensor<128x1xi64>
    %72 = linalg.fill ins(%28 : i64) outs(%71 : tensor<128x1xi64>) -> tensor<128x1xi64>
    %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%70 : tensor<128x1xi32>) outs(%71 : tensor<128x1xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<128x1xi64>
    %74 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%72, %73 : tensor<128x1xi64>, tensor<128x1xi64>) outs(%72 : tensor<128x1xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<128x1xi64>
    %expanded_9 = tensor.expand_shape %61 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %75 = tensor.empty() : tensor<128x16xi64>
    %76 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%74 : tensor<128x1xi64>) outs(%75 : tensor<128x16xi64>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<128x16xi64>
    %77 = tensor.empty() : tensor<128x16xi32>
    %78 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_9 : tensor<1x16xi32>) outs(%77 : tensor<128x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<128x16xi32>
    %79 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%78 : tensor<128x16xi32>) outs(%75 : tensor<128x16xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<128x16xi64>
    %80 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%76, %79 : tensor<128x16xi64>, tensor<128x16xi64>) outs(%76 : tensor<128x16xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<128x16xi64>
    %expanded_10 = tensor.expand_shape %59 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %81 = linalg.fill ins(%arg23 : i32) outs(%12 : tensor<1x64xi32>) -> tensor<1x64xi32>
    %82 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10, %81 : tensor<1x64xi32>, tensor<1x64xi32>) outs(%expanded_10 : tensor<1x64xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.muli %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<1x64xi32>
    %83 = tensor.empty() : tensor<1x64xi64>
    %84 = linalg.fill ins(%54 : i64) outs(%83 : tensor<1x64xi64>) -> tensor<1x64xi64>
    %85 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%82 : tensor<1x64xi32>) outs(%83 : tensor<1x64xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<1x64xi64>
    %86 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%84, %85 : tensor<1x64xi64>, tensor<1x64xi64>) outs(%84 : tensor<1x64xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<1x64xi64>
    %expanded_11 = tensor.expand_shape %61 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
    %87 = tensor.empty() : tensor<16x64xi64>
    %88 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%86 : tensor<1x64xi64>) outs(%87 : tensor<16x64xi64>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<16x64xi64>
    %89 = tensor.empty() : tensor<16x64xi32>
    %90 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<16x1xi32>) outs(%89 : tensor<16x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<16x64xi32>
    %91 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%90 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %92 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%88, %91 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%88 : tensor<16x64xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %expanded_12 = tensor.expand_shape %59 [[0, 1]] output_shape [64, 1] : tensor<64xi32> into tensor<64x1xi32>
    %93 = tensor.empty() : tensor<64x1xi32>
    %94 = linalg.fill ins(%arg23 : i32) outs(%93 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %95 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12, %94 : tensor<64x1xi32>, tensor<64x1xi32>) outs(%expanded_12 : tensor<64x1xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.muli %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<64x1xi32>
    %96 = tensor.empty() : tensor<64x1xi64>
    %97 = linalg.fill ins(%54 : i64) outs(%96 : tensor<64x1xi64>) -> tensor<64x1xi64>
    %98 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%95 : tensor<64x1xi32>) outs(%96 : tensor<64x1xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<64x1xi64>
    %99 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%97, %98 : tensor<64x1xi64>, tensor<64x1xi64>) outs(%97 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<64x1xi64>
    %100 = tensor.empty() : tensor<64x16xi64>
    %101 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%99 : tensor<64x1xi64>) outs(%100 : tensor<64x16xi64>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<64x16xi64>
    %102 = tensor.empty() : tensor<64x16xi32>
    %103 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_9 : tensor<1x16xi32>) outs(%102 : tensor<64x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<64x16xi32>
    %104 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%103 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %105 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%101, %104 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%101 : tensor<64x16xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %106 = arith.mulf %arg5, %cst_2 : f32
    %cast = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %107 = bufferization.to_tensor %cast restrict : memref<?xf32>
    %108 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%80 : tensor<128x16xi64>) outs(%4 : tensor<128x16xf32>) {
    ^bb0(%in: i64, %out: f32):
      %208 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %107[%208] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x16xf32>
    %109 = tensor.empty() : tensor<128xi64>
    %110 = linalg.fill ins(%62 : i64) outs(%109 : tensor<128xi64>) -> tensor<128xi64>
    %111 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%47 : tensor<128xi32>) outs(%109 : tensor<128xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<128xi64>
    %112 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%110, %111 : tensor<128xi64>, tensor<128xi64>) outs(%110 : tensor<128xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<128xi64>
    %cast_13 = memref.cast %arg14 : memref<*xi32> to memref<?xi32>
    %113 = bufferization.to_tensor %cast_13 restrict : memref<?xi32>
    %114 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%112 : tensor<128xi64>) outs(%0 : tensor<128xi32>) {
    ^bb0(%in: i64, %out: i32):
      %208 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %113[%208] : tensor<?xi32>
      linalg.yield %extracted : i32
    } -> tensor<128xi32>
    %115 = arith.muli %arg38, %c6_i32 : i32
    %116 = arith.addi %115, %c1_i32 : i32
    %117 = arith.index_cast %115 : i32 to index
    %reinterpret_cast_14 = memref.reinterpret_cast %arg17 to offset: [%117], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %118 = affine.load %reinterpret_cast_14[0] : memref<1xi32, strided<[1], offset: ?>>
    %119 = arith.index_cast %116 : i32 to index
    %reinterpret_cast_15 = memref.reinterpret_cast %arg17 to offset: [%119], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %120 = affine.load %reinterpret_cast_15[0] : memref<1xi32, strided<[1], offset: ?>>
    %121 = arith.muli %118, %arg23 : i32
    %122 = linalg.fill ins(%121 : i32) outs(%89 : tensor<16x64xi32>) -> tensor<16x64xi32>
    %123 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%122 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %124 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%92, %123 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%92 : tensor<16x64xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %125 = linalg.fill ins(%121 : i32) outs(%102 : tensor<64x16xi32>) -> tensor<64x16xi32>
    %126 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%125 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %127 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%105, %126 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%105 : tensor<64x16xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %128 = arith.subi %120, %118 : i32
    %expanded_16 = tensor.expand_shape %114 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %129 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_16 : tensor<128x1xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<128x64xi32>
    %expanded_17 = tensor.expand_shape %48 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %130 = linalg.fill ins(%arg32 : i32) outs(%68 : tensor<128x1xi32>) -> tensor<128x1xi32>
    %131 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_17, %130 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%expanded_17 : tensor<128x1xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.muli %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128x1xi32>
    %132 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%131 : tensor<128x1xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<128x64xi32>
    %expanded_18 = tensor.expand_shape %47 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %133 = linalg.fill ins(%arg33 : i32) outs(%68 : tensor<128x1xi32>) -> tensor<128x1xi32>
    %134 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_18, %133 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%expanded_18 : tensor<128x1xi32>) {
    ^bb0(%in: i32, %in_37: i32, %out: i32):
      %208 = arith.muli %in, %in_37 : i32
      linalg.yield %208 : i32
    } -> tensor<128x1xi32>
    %135 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%134 : tensor<128x1xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<128x64xi32>
    %136 = tensor.empty() : tensor<128x64xi64>
    %137 = linalg.fill ins(%67 : i64) outs(%136 : tensor<128x64xi64>) -> tensor<128x64xi64>
    %138 = linalg.fill ins(%106 : f32) outs(%6 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %139 = arith.muli %arg23, %c64_i32 : i32
    %140 = linalg.fill ins(%139 : i32) outs(%102 : tensor<64x16xi32>) -> tensor<64x16xi32>
    %141 = linalg.fill ins(%139 : i32) outs(%89 : tensor<16x64xi32>) -> tensor<16x64xi32>
    %142:5 = scf.for %arg40 = %c0_i32 to %128 step %c64_i32 iter_args(%arg41 = %3, %arg42 = %5, %arg43 = %15, %arg44 = %127, %arg45 = %124) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %208 = arith.addi %118, %arg40 : i32
      %209 = linalg.fill ins(%208 : i32) outs(%57 : tensor<64xi32>) -> tensor<64xi32>
      %210 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%209, %58 : tensor<64xi32>, tensor<64xi32>) outs(%209 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.addi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<64xi32>
      %cast_37 = memref.cast %arg3 : memref<*xf32> to memref<?xf32>
      %211 = bufferization.to_tensor %cast_37 restrict : memref<?xf32>
      %212 = tensor.empty() : tensor<16x64xf32>
      %213 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45 : tensor<16x64xi64>) outs(%212 : tensor<16x64xf32>) {
      ^bb0(%in: i64, %out: f32):
        %256 = arith.index_cast %in : i64 to index
        %extracted = tensor.extract %211[%256] : tensor<?xf32>
        linalg.yield %extracted : f32
      } -> tensor<16x64xf32>
      %expanded_38 = tensor.expand_shape %210 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
      %214 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_38, %13 : tensor<1x64xi32>, tensor<1x64xi32>) outs(%expanded_38 : tensor<1x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.addi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<1x64xi32>
      %215 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%214 : tensor<1x64xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<128x64xi32>
      %216 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%215, %129 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%215 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.subi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<128x64xi32>
      %217 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%216, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%216 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.subi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<128x64xi32>
      %218 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%217, %132 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%217 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.addi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<128x64xi32>
      %219 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%218, %135 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%218 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.addi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<128x64xi32>
      %220 = tensor.empty() : tensor<128x64xi1>
      %221 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%217, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%220 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_45: i32, %out: i1):
        %256 = arith.cmpi ult, %in, %in_45 : i32
        linalg.yield %256 : i1
      } -> tensor<128x64xi1>
      %222 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%219 : tensor<128x64xi32>) outs(%136 : tensor<128x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %256 = arith.extsi %in : i32 to i64
        linalg.yield %256 : i64
      } -> tensor<128x64xi64>
      %223 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%137, %222 : tensor<128x64xi64>, tensor<128x64xi64>) outs(%137 : tensor<128x64xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %256 = arith.addi %in, %in_45 : i64
        linalg.yield %256 : i64
      } -> tensor<128x64xi64>
      %cast_39 = memref.cast %arg30 : memref<*xf32> to memref<?xf32>
      %224 = bufferization.to_tensor %cast_39 restrict : memref<?xf32>
      %225 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%223, %221 : tensor<128x64xi64>, tensor<128x64xi1>) outs(%6 : tensor<128x64xf32>) {
      ^bb0(%in: i64, %in_45: i1, %out: f32):
        %256 = scf.if %in_45 -> (f32) {
          %257 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %224[%257] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_1 : f32
        }
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %226 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%221, %225, %10 : tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) outs(%225 : tensor<128x64xf32>) {
      ^bb0(%in: i1, %in_45: f32, %in_46: f32, %out: f32):
        %256 = arith.select %in, %in_45, %in_46 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %227 = linalg.matmul ins(%108, %213 : tensor<128x16xf32>, tensor<16x64xf32>) outs(%10 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %228 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%227, %226 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%227 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.addf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %229 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_38 : tensor<1x64xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<128x64xi32>
      %230 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%129, %229 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%129 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %256 = arith.subi %in, %in_45 : i32
        linalg.yield %256 : i32
      } -> tensor<128x64xi32>
      %231 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%230, %9 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%220 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_45: i32, %out: i1):
        %256 = arith.cmpi sge, %in, %in_45 : i32
        linalg.yield %256 : i1
      } -> tensor<128x64xi1>
      %232 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%230, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%220 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_45: i32, %out: i1):
        %256 = arith.cmpi slt, %in, %in_45 : i32
        linalg.yield %256 : i1
      } -> tensor<128x64xi1>
      %233 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%231, %232 : tensor<128x64xi1>, tensor<128x64xi1>) outs(%231 : tensor<128x64xi1>) {
      ^bb0(%in: i1, %in_45: i1, %out: i1):
        %256 = arith.andi %in, %in_45 : i1
        linalg.yield %256 : i1
      } -> tensor<128x64xi1>
      %234 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%228, %138 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%228 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.mulf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %235 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%233, %10, %7 : tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) outs(%10 : tensor<128x64xf32>) {
      ^bb0(%in: i1, %in_45: f32, %in_46: f32, %out: f32):
        %256 = arith.select %in, %in_45, %in_46 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %236 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%234, %235 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%234 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.addf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %237 = tensor.empty() : tensor<64x128xf32>
      %transposed = linalg.transpose ins(%236 : tensor<128x64xf32>) outs(%237 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced = linalg.reduce ins(%transposed : tensor<64x128xf32>) outs(%15 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %256 = arith.maxnumf %in, %init : f32
          linalg.yield %256 : f32
        }
      %238 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %reduced : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.maxnumf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128xf32>
      %expanded_40 = tensor.expand_shape %238 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %239 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_40 : tensor<128x1xf32>) outs(%6 : tensor<128x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x64xf32>
      %240 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%236, %239 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%236 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.subf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %241 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%240 : tensor<128x64xf32>) outs(%240 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %256 = math.exp2 %in : f32
        linalg.yield %256 : f32
      } -> tensor<128x64xf32>
      %transposed_41 = linalg.transpose ins(%241 : tensor<128x64xf32>) outs(%237 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced_42 = linalg.reduce ins(%transposed_41 : tensor<64x128xf32>) outs(%3 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %256 = arith.addf %in, %init : f32
          linalg.yield %256 : f32
        }
      %242 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %238 : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.subf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128xf32>
      %243 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%242 : tensor<128xf32>) outs(%242 : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %256 = math.exp2 %in : f32
        linalg.yield %256 : f32
      } -> tensor<128xf32>
      %244 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg41, %243 : tensor<128xf32>, tensor<128xf32>) outs(%arg41 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.mulf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128xf32>
      %245 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%244, %reduced_42 : tensor<128xf32>, tensor<128xf32>) outs(%244 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.addf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128xf32>
      %expanded_43 = tensor.expand_shape %243 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %246 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_43 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x16xf32>
      %247 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg42, %246 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%arg42 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.mulf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x16xf32>
      %cast_44 = memref.cast %arg4 : memref<*xf32> to memref<?xf32>
      %248 = bufferization.to_tensor %cast_44 restrict : memref<?xf32>
      %249 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44 : tensor<64x16xi64>) outs(%14 : tensor<64x16xf32>) {
      ^bb0(%in: i64, %out: f32):
        %256 = arith.index_cast %in : i64 to index
        %extracted = tensor.extract %248[%256] : tensor<?xf32>
        linalg.yield %extracted : f32
      } -> tensor<64x16xf32>
      %250 = linalg.matmul ins(%241, %249 : tensor<128x64xf32>, tensor<64x16xf32>) outs(%5 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %251 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%250, %247 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%250 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %256 = arith.addf %in, %in_45 : f32
        linalg.yield %256 : f32
      } -> tensor<128x16xf32>
      %252 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%140 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
      ^bb0(%in: i32, %out: i64):
        %256 = arith.extsi %in : i32 to i64
        linalg.yield %256 : i64
      } -> tensor<64x16xi64>
      %253 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44, %252 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%arg44 : tensor<64x16xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %256 = arith.addi %in, %in_45 : i64
        linalg.yield %256 : i64
      } -> tensor<64x16xi64>
      %254 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%141 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %256 = arith.extsi %in : i32 to i64
        linalg.yield %256 : i64
      } -> tensor<16x64xi64>
      %255 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45, %254 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%arg45 : tensor<16x64xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %256 = arith.addi %in, %in_45 : i64
        linalg.yield %256 : i64
      } -> tensor<16x64xi64>
      scf.yield %245, %251, %238, %253, %255 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %143 = arith.addi %115, %c2_i32 : i32
    %144 = arith.addi %115, %c3_i32 : i32
    %145 = arith.index_cast %143 : i32 to index
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [%145], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %146 = affine.load %reinterpret_cast_19[0] : memref<1xi32, strided<[1], offset: ?>>
    %147 = arith.index_cast %144 : i32 to index
    %reinterpret_cast_20 = memref.reinterpret_cast %arg17 to offset: [%147], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %148 = affine.load %reinterpret_cast_20[0] : memref<1xi32, strided<[1], offset: ?>>
    %149 = arith.muli %146, %arg23 : i32
    %150 = linalg.fill ins(%149 : i32) outs(%89 : tensor<16x64xi32>) -> tensor<16x64xi32>
    %151 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%150 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %152 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%92, %151 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%92 : tensor<16x64xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %153 = linalg.fill ins(%149 : i32) outs(%102 : tensor<64x16xi32>) -> tensor<64x16xi32>
    %154 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%153 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%105, %154 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%105 : tensor<64x16xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %156 = arith.subi %148, %146 : i32
    %157 = linalg.fill ins(%106 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %158:5 = scf.for %arg40 = %c0_i32 to %156 step %c64_i32 iter_args(%arg41 = %142#0, %arg42 = %142#1, %arg43 = %142#2, %arg44 = %155, %arg45 = %152) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %208 = arith.addi %146, %arg40 : i32
      %209 = linalg.fill ins(%208 : i32) outs(%57 : tensor<64xi32>) -> tensor<64xi32>
      %210 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%209, %58 : tensor<64xi32>, tensor<64xi32>) outs(%209 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.addi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<64xi32>
      %cast_37 = memref.cast %arg3 : memref<*xf32> to memref<?xf32>
      %211 = bufferization.to_tensor %cast_37 restrict : memref<?xf32>
      %212 = tensor.empty() : tensor<16x64xf32>
      %213 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45 : tensor<16x64xi64>) outs(%212 : tensor<16x64xf32>) {
      ^bb0(%in: i64, %out: f32):
        %250 = arith.index_cast %in : i64 to index
        %extracted = tensor.extract %211[%250] : tensor<?xf32>
        linalg.yield %extracted : f32
      } -> tensor<16x64xf32>
      %expanded_38 = tensor.expand_shape %210 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
      %214 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_38, %13 : tensor<1x64xi32>, tensor<1x64xi32>) outs(%expanded_38 : tensor<1x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.addi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<1x64xi32>
      %215 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%214 : tensor<1x64xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<128x64xi32>
      %216 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%215, %129 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%215 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.subi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<128x64xi32>
      %217 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%216, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%216 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.subi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<128x64xi32>
      %218 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%217, %132 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%217 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.addi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<128x64xi32>
      %219 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%218, %135 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%218 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_45: i32, %out: i32):
        %250 = arith.addi %in, %in_45 : i32
        linalg.yield %250 : i32
      } -> tensor<128x64xi32>
      %220 = tensor.empty() : tensor<128x64xi1>
      %221 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%217, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%220 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_45: i32, %out: i1):
        %250 = arith.cmpi ult, %in, %in_45 : i32
        linalg.yield %250 : i1
      } -> tensor<128x64xi1>
      %222 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%219 : tensor<128x64xi32>) outs(%136 : tensor<128x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %250 = arith.extsi %in : i32 to i64
        linalg.yield %250 : i64
      } -> tensor<128x64xi64>
      %223 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%137, %222 : tensor<128x64xi64>, tensor<128x64xi64>) outs(%137 : tensor<128x64xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %250 = arith.addi %in, %in_45 : i64
        linalg.yield %250 : i64
      } -> tensor<128x64xi64>
      %cast_39 = memref.cast %arg30 : memref<*xf32> to memref<?xf32>
      %224 = bufferization.to_tensor %cast_39 restrict : memref<?xf32>
      %225 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%223, %221 : tensor<128x64xi64>, tensor<128x64xi1>) outs(%6 : tensor<128x64xf32>) {
      ^bb0(%in: i64, %in_45: i1, %out: f32):
        %250 = scf.if %in_45 -> (f32) {
          %251 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %224[%251] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_1 : f32
        }
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %226 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%221, %225, %10 : tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) outs(%225 : tensor<128x64xf32>) {
      ^bb0(%in: i1, %in_45: f32, %in_46: f32, %out: f32):
        %250 = arith.select %in, %in_45, %in_46 : f32
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %227 = linalg.matmul ins(%108, %213 : tensor<128x16xf32>, tensor<16x64xf32>) outs(%10 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %228 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%227, %226 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%227 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.addf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %229 = tensor.empty() : tensor<64x128xf32>
      %transposed = linalg.transpose ins(%228 : tensor<128x64xf32>) outs(%229 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced = linalg.reduce ins(%transposed : tensor<64x128xf32>) outs(%15 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %250 = arith.maxnumf %in, %init : f32
          linalg.yield %250 : f32
        }
      %230 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reduced, %157 : tensor<128xf32>, tensor<128xf32>) outs(%reduced : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.mulf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %231 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %230 : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.maxnumf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %232 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%228, %138 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%228 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.mulf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %expanded_40 = tensor.expand_shape %231 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %233 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_40 : tensor<128x1xf32>) outs(%6 : tensor<128x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x64xf32>
      %234 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%232, %233 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%232 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.subf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %235 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%234 : tensor<128x64xf32>) outs(%234 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %250 = math.exp2 %in : f32
        linalg.yield %250 : f32
      } -> tensor<128x64xf32>
      %transposed_41 = linalg.transpose ins(%235 : tensor<128x64xf32>) outs(%229 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced_42 = linalg.reduce ins(%transposed_41 : tensor<64x128xf32>) outs(%3 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %250 = arith.addf %in, %init : f32
          linalg.yield %250 : f32
        }
      %236 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %231 : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.subf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %237 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%236 : tensor<128xf32>) outs(%236 : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %250 = math.exp2 %in : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %238 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg41, %237 : tensor<128xf32>, tensor<128xf32>) outs(%arg41 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.mulf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %239 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%238, %reduced_42 : tensor<128xf32>, tensor<128xf32>) outs(%238 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.addf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128xf32>
      %expanded_43 = tensor.expand_shape %237 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %240 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_43 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x16xf32>
      %241 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg42, %240 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%arg42 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.mulf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128x16xf32>
      %cast_44 = memref.cast %arg4 : memref<*xf32> to memref<?xf32>
      %242 = bufferization.to_tensor %cast_44 restrict : memref<?xf32>
      %243 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44 : tensor<64x16xi64>) outs(%14 : tensor<64x16xf32>) {
      ^bb0(%in: i64, %out: f32):
        %250 = arith.index_cast %in : i64 to index
        %extracted = tensor.extract %242[%250] : tensor<?xf32>
        linalg.yield %extracted : f32
      } -> tensor<64x16xf32>
      %244 = linalg.matmul ins(%235, %243 : tensor<128x64xf32>, tensor<64x16xf32>) outs(%5 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %245 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%244, %241 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%244 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_45: f32, %out: f32):
        %250 = arith.addf %in, %in_45 : f32
        linalg.yield %250 : f32
      } -> tensor<128x16xf32>
      %246 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%140 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
      ^bb0(%in: i32, %out: i64):
        %250 = arith.extsi %in : i32 to i64
        linalg.yield %250 : i64
      } -> tensor<64x16xi64>
      %247 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44, %246 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%arg44 : tensor<64x16xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %250 = arith.addi %in, %in_45 : i64
        linalg.yield %250 : i64
      } -> tensor<64x16xi64>
      %248 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%141 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %250 = arith.extsi %in : i32 to i64
        linalg.yield %250 : i64
      } -> tensor<16x64xi64>
      %249 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45, %248 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%arg45 : tensor<16x64xi64>) {
      ^bb0(%in: i64, %in_45: i64, %out: i64):
        %250 = arith.addi %in, %in_45 : i64
        linalg.yield %250 : i64
      } -> tensor<16x64xi64>
      scf.yield %239, %245, %231, %247, %249 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %159 = arith.addi %115, %c4_i32 : i32
    %160 = arith.addi %115, %c5_i32 : i32
    %161 = arith.index_cast %159 : i32 to index
    %reinterpret_cast_21 = memref.reinterpret_cast %arg17 to offset: [%161], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %162 = affine.load %reinterpret_cast_21[0] : memref<1xi32, strided<[1], offset: ?>>
    %163 = arith.index_cast %160 : i32 to index
    %reinterpret_cast_22 = memref.reinterpret_cast %arg17 to offset: [%163], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %164 = affine.load %reinterpret_cast_22[0] : memref<1xi32, strided<[1], offset: ?>>
    %165 = arith.muli %162, %arg23 : i32
    %166 = linalg.fill ins(%165 : i32) outs(%89 : tensor<16x64xi32>) -> tensor<16x64xi32>
    %167 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%166 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %168 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%92, %167 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%92 : tensor<16x64xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<16x64xi64>
    %169 = linalg.fill ins(%165 : i32) outs(%102 : tensor<64x16xi32>) -> tensor<64x16xi32>
    %170 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%169 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
    ^bb0(%in: i32, %out: i64):
      %208 = arith.extsi %in : i32 to i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %171 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%105, %170 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%105 : tensor<64x16xi64>) {
    ^bb0(%in: i64, %in_37: i64, %out: i64):
      %208 = arith.addi %in, %in_37 : i64
      linalg.yield %208 : i64
    } -> tensor<64x16xi64>
    %172 = arith.subi %164, %162 : i32
    %173 = linalg.fill ins(%49 : i32) outs(%57 : tensor<64xi32>) -> tensor<64xi32>
    %174:5 = scf.for %arg40 = %c0_i32 to %172 step %c64_i32 iter_args(%arg41 = %158#0, %arg42 = %158#1, %arg43 = %158#2, %arg44 = %171, %arg45 = %168) -> (tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>)  : i32 {
      %208 = arith.addi %162, %arg40 : i32
      %209 = linalg.fill ins(%208 : i32) outs(%57 : tensor<64xi32>) -> tensor<64xi32>
      %210 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%209, %58 : tensor<64xi32>, tensor<64xi32>) outs(%209 : tensor<64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.addi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<64xi32>
      %211 = tensor.empty() : tensor<64xi1>
      %212 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%210, %173 : tensor<64xi32>, tensor<64xi32>) outs(%211 : tensor<64xi1>) {
      ^bb0(%in: i32, %in_47: i32, %out: i1):
        %264 = arith.cmpi slt, %in, %in_47 : i32
        linalg.yield %264 : i1
      } -> tensor<64xi1>
      %expanded_37 = tensor.expand_shape %212 [[0, 1]] output_shape [1, 64] : tensor<64xi1> into tensor<1x64xi1>
      %213 = tensor.empty() : tensor<16x64xi1>
      %214 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_37 : tensor<1x64xi1>) outs(%213 : tensor<16x64xi1>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i1, %out: i1):
        linalg.yield %in : i1
      } -> tensor<16x64xi1>
      %cast_38 = memref.cast %arg3 : memref<*xf32> to memref<?xf32>
      %215 = bufferization.to_tensor %cast_38 restrict : memref<?xf32>
      %216 = tensor.empty() : tensor<16x64xf32>
      %217 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45, %214 : tensor<16x64xi64>, tensor<16x64xi1>) outs(%216 : tensor<16x64xf32>) {
      ^bb0(%in: i64, %in_47: i1, %out: f32):
        %264 = scf.if %in_47 -> (f32) {
          %265 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %215[%265] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_1 : f32
        }
        linalg.yield %264 : f32
      } -> tensor<16x64xf32>
      %expanded_39 = tensor.expand_shape %210 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
      %218 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_39, %13 : tensor<1x64xi32>, tensor<1x64xi32>) outs(%expanded_39 : tensor<1x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.addi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<1x64xi32>
      %219 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%218 : tensor<1x64xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<128x64xi32>
      %220 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%219, %129 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%219 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.subi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<128x64xi32>
      %221 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%220, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%220 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.subi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<128x64xi32>
      %222 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%221, %132 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%221 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.addi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<128x64xi32>
      %223 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%222, %135 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%222 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.addi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<128x64xi32>
      %224 = tensor.empty() : tensor<128x64xi1>
      %225 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%221, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%224 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_47: i32, %out: i1):
        %264 = arith.cmpi ult, %in, %in_47 : i32
        linalg.yield %264 : i1
      } -> tensor<128x64xi1>
      %226 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%223 : tensor<128x64xi32>) outs(%136 : tensor<128x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %264 = arith.extsi %in : i32 to i64
        linalg.yield %264 : i64
      } -> tensor<128x64xi64>
      %227 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%137, %226 : tensor<128x64xi64>, tensor<128x64xi64>) outs(%137 : tensor<128x64xi64>) {
      ^bb0(%in: i64, %in_47: i64, %out: i64):
        %264 = arith.addi %in, %in_47 : i64
        linalg.yield %264 : i64
      } -> tensor<128x64xi64>
      %cast_40 = memref.cast %arg30 : memref<*xf32> to memref<?xf32>
      %228 = bufferization.to_tensor %cast_40 restrict : memref<?xf32>
      %229 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%227, %225 : tensor<128x64xi64>, tensor<128x64xi1>) outs(%6 : tensor<128x64xf32>) {
      ^bb0(%in: i64, %in_47: i1, %out: f32):
        %264 = scf.if %in_47 -> (f32) {
          %265 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %228[%265] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_1 : f32
        }
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %230 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%225, %229, %10 : tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) outs(%229 : tensor<128x64xf32>) {
      ^bb0(%in: i1, %in_47: f32, %in_48: f32, %out: f32):
        %264 = arith.select %in, %in_47, %in_48 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %231 = linalg.matmul ins(%108, %217 : tensor<128x16xf32>, tensor<16x64xf32>) outs(%10 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %232 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%231, %230 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%231 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.addf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %233 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_39 : tensor<1x64xi32>) outs(%8 : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<128x64xi32>
      %234 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%129, %233 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%129 : tensor<128x64xi32>) {
      ^bb0(%in: i32, %in_47: i32, %out: i32):
        %264 = arith.subi %in, %in_47 : i32
        linalg.yield %264 : i32
      } -> tensor<128x64xi32>
      %235 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%234, %9 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%224 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_47: i32, %out: i1):
        %264 = arith.cmpi sge, %in, %in_47 : i32
        linalg.yield %264 : i1
      } -> tensor<128x64xi1>
      %236 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%234, %11 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%224 : tensor<128x64xi1>) {
      ^bb0(%in: i32, %in_47: i32, %out: i1):
        %264 = arith.cmpi slt, %in, %in_47 : i32
        linalg.yield %264 : i1
      } -> tensor<128x64xi1>
      %237 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%235, %236 : tensor<128x64xi1>, tensor<128x64xi1>) outs(%235 : tensor<128x64xi1>) {
      ^bb0(%in: i1, %in_47: i1, %out: i1):
        %264 = arith.andi %in, %in_47 : i1
        linalg.yield %264 : i1
      } -> tensor<128x64xi1>
      %238 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_37 : tensor<1x64xi1>) outs(%224 : tensor<128x64xi1>) attrs =  {broadcastDims = array<i64: 0>} {
      ^bb0(%in: i1, %out: i1):
        linalg.yield %in : i1
      } -> tensor<128x64xi1>
      %239 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%237, %238 : tensor<128x64xi1>, tensor<128x64xi1>) outs(%237 : tensor<128x64xi1>) {
      ^bb0(%in: i1, %in_47: i1, %out: i1):
        %264 = arith.andi %in, %in_47 : i1
        linalg.yield %264 : i1
      } -> tensor<128x64xi1>
      %240 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%232, %138 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%232 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.mulf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %241 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%239, %10, %7 : tensor<128x64xi1>, tensor<128x64xf32>, tensor<128x64xf32>) outs(%10 : tensor<128x64xf32>) {
      ^bb0(%in: i1, %in_47: f32, %in_48: f32, %out: f32):
        %264 = arith.select %in, %in_47, %in_48 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %242 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%240, %241 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%240 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.addf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %243 = tensor.empty() : tensor<64x128xf32>
      %transposed = linalg.transpose ins(%242 : tensor<128x64xf32>) outs(%243 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced = linalg.reduce ins(%transposed : tensor<64x128xf32>) outs(%15 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %264 = arith.maxnumf %in, %init : f32
          linalg.yield %264 : f32
        }
      %244 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %reduced : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.maxnumf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128xf32>
      %expanded_41 = tensor.expand_shape %244 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %245 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_41 : tensor<128x1xf32>) outs(%6 : tensor<128x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x64xf32>
      %246 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%242, %245 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%242 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.subf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %247 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%246 : tensor<128x64xf32>) outs(%246 : tensor<128x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %264 = math.exp2 %in : f32
        linalg.yield %264 : f32
      } -> tensor<128x64xf32>
      %transposed_42 = linalg.transpose ins(%247 : tensor<128x64xf32>) outs(%243 : tensor<64x128xf32>) permutation = [1, 0]
      %reduced_43 = linalg.reduce ins(%transposed_42 : tensor<64x128xf32>) outs(%3 : tensor<128xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %264 = arith.addf %in, %init : f32
          linalg.yield %264 : f32
        }
      %248 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg43, %244 : tensor<128xf32>, tensor<128xf32>) outs(%arg43 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.subf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128xf32>
      %249 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%248 : tensor<128xf32>) outs(%248 : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %264 = math.exp2 %in : f32
        linalg.yield %264 : f32
      } -> tensor<128xf32>
      %250 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg41, %249 : tensor<128xf32>, tensor<128xf32>) outs(%arg41 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.mulf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128xf32>
      %251 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%250, %reduced_43 : tensor<128xf32>, tensor<128xf32>) outs(%250 : tensor<128xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.addf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128xf32>
      %expanded_44 = tensor.expand_shape %249 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %252 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_44 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<128x16xf32>
      %253 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg42, %252 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%arg42 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.mulf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x16xf32>
      %expanded_45 = tensor.expand_shape %212 [[0, 1]] output_shape [64, 1] : tensor<64xi1> into tensor<64x1xi1>
      %254 = tensor.empty() : tensor<64x16xi1>
      %255 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_45 : tensor<64x1xi1>) outs(%254 : tensor<64x16xi1>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: i1, %out: i1):
        linalg.yield %in : i1
      } -> tensor<64x16xi1>
      %cast_46 = memref.cast %arg4 : memref<*xf32> to memref<?xf32>
      %256 = bufferization.to_tensor %cast_46 restrict : memref<?xf32>
      %257 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44, %255 : tensor<64x16xi64>, tensor<64x16xi1>) outs(%14 : tensor<64x16xf32>) {
      ^bb0(%in: i64, %in_47: i1, %out: f32):
        %264 = scf.if %in_47 -> (f32) {
          %265 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %256[%265] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst_1 : f32
        }
        linalg.yield %264 : f32
      } -> tensor<64x16xf32>
      %258 = linalg.matmul ins(%247, %257 : tensor<128x64xf32>, tensor<64x16xf32>) outs(%5 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %259 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%258, %253 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%258 : tensor<128x16xf32>) {
      ^bb0(%in: f32, %in_47: f32, %out: f32):
        %264 = arith.addf %in, %in_47 : f32
        linalg.yield %264 : f32
      } -> tensor<128x16xf32>
      %260 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%140 : tensor<64x16xi32>) outs(%100 : tensor<64x16xi64>) {
      ^bb0(%in: i32, %out: i64):
        %264 = arith.extsi %in : i32 to i64
        linalg.yield %264 : i64
      } -> tensor<64x16xi64>
      %261 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg44, %260 : tensor<64x16xi64>, tensor<64x16xi64>) outs(%arg44 : tensor<64x16xi64>) {
      ^bb0(%in: i64, %in_47: i64, %out: i64):
        %264 = arith.addi %in, %in_47 : i64
        linalg.yield %264 : i64
      } -> tensor<64x16xi64>
      %262 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%141 : tensor<16x64xi32>) outs(%87 : tensor<16x64xi64>) {
      ^bb0(%in: i32, %out: i64):
        %264 = arith.extsi %in : i32 to i64
        linalg.yield %264 : i64
      } -> tensor<16x64xi64>
      %263 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg45, %262 : tensor<16x64xi64>, tensor<16x64xi64>) outs(%arg45 : tensor<16x64xi64>) {
      ^bb0(%in: i64, %in_47: i64, %out: i64):
        %264 = arith.addi %in, %in_47 : i64
        linalg.yield %264 : i64
      } -> tensor<16x64xi64>
      scf.yield %251, %259, %244, %261, %263 : tensor<128xf32>, tensor<128x16xf32>, tensor<128xf32>, tensor<64x16xi64>, tensor<16x64xi64>
    }
    %175 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%174#0 : tensor<128xf32>) outs(%174#0 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.log2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %176 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%174#2, %175 : tensor<128xf32>, tensor<128xf32>) outs(%174#2 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.addf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %expanded_23 = tensor.expand_shape %174#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %177 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_23 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x16xf32>
    %178 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%174#1, %177 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%174#1 : tensor<128x16xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.divf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128x16xf32>
    %179 = arith.index_cast %arg20 : i32 to index
    %180 = arith.muli %37, %179 : index
    %181 = arith.addi %29, %180 : index
    %reinterpret_cast_24 = memref.reinterpret_cast %arg6 to offset: [%181], sizes: [128, 16], strides: [%179, 1] : memref<*xf32> to memref<128x16xf32, strided<[?, 1], offset: ?>>
    %182 = arith.addi %33, %37 : index
    %reinterpret_cast_25 = memref.reinterpret_cast %arg7 to offset: [%182], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %183 = arith.addi %37, %c128 : index
    %184 = arith.index_cast %42 : i32 to index
    %185 = arith.minsi %183, %184 : index
    %186 = arith.subi %185, %37 : index
    %alloc_26 = memref.alloc() : memref<128xf32>
    %subview = memref.subview %reinterpret_cast_25[0] [%186] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_27 = memref.subview %alloc_26[0] [%186] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_27 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %187 = bufferization.to_tensor %alloc_26 restrict writable : memref<128xf32>
    %expanded_28 = tensor.expand_shape %45 [[0, 1]] output_shape [128, 1] : tensor<128xi1> into tensor<128x1xi1>
    %188 = tensor.empty() : tensor<128x16xi1>
    %189 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_28 : tensor<128x1xi1>) outs(%188 : tensor<128x16xi1>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i1, %out: i1):
      linalg.yield %in : i1
    } -> tensor<128x16xi1>
    %alloc_29 = memref.alloc() : memref<128x16xf32>
    %subview_30 = memref.subview %reinterpret_cast_24[0, 0] [%186, 16] [1, 1] : memref<128x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[?, 1], offset: ?>>
    %subview_31 = memref.subview %alloc_29[0, 0] [%186, 16] [1, 1] : memref<128x16xf32> to memref<?x16xf32, strided<[16, 1]>>
    memref.copy %subview_30, %subview_31 : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1]>>
    %190 = bufferization.to_tensor %alloc_29 restrict writable : memref<128x16xf32>
    %191 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%187, %176 : tensor<128xf32>, tensor<128xf32>) outs(%187 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.maxnumf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %192 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%176, %191 : tensor<128xf32>, tensor<128xf32>) outs(%176 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.subf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %193 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%192 : tensor<128xf32>) outs(%192 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.exp2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %194 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%187, %191 : tensor<128xf32>, tensor<128xf32>) outs(%187 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.subf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %195 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%194 : tensor<128xf32>) outs(%194 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.exp2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %196 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%193, %195 : tensor<128xf32>, tensor<128xf32>) outs(%193 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.addf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %197 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%196 : tensor<128xf32>) outs(%196 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.log2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %198 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%191, %197 : tensor<128xf32>, tensor<128xf32>) outs(%191 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.addf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %199 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%176, %198 : tensor<128xf32>, tensor<128xf32>) outs(%176 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.subf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %200 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%199 : tensor<128xf32>) outs(%199 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.exp2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %expanded_32 = tensor.expand_shape %200 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %201 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_32 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x16xf32>
    %202 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%178, %201 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%178 : tensor<128x16xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.mulf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128x16xf32>
    %203 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%187, %198 : tensor<128xf32>, tensor<128xf32>) outs(%187 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.subf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %204 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%203 : tensor<128xf32>) outs(%203 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %208 = math.exp2 %in : f32
      linalg.yield %208 : f32
    } -> tensor<128xf32>
    %expanded_33 = tensor.expand_shape %204 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %205 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_33 : tensor<128x1xf32>) outs(%4 : tensor<128x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x16xf32>
    %206 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%190, %205 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%190 : tensor<128x16xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.mulf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128x16xf32>
    %207 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%202, %206 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%202 : tensor<128x16xf32>) {
    ^bb0(%in: f32, %in_37: f32, %out: f32):
      %208 = arith.addf %in, %in_37 : f32
      linalg.yield %208 : f32
    } -> tensor<128x16xf32>
    %reinterpret_cast_34 = memref.reinterpret_cast %arg1 to offset: [%182], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %198[0] [%186] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_35 = memref.subview %reinterpret_cast_34[0] [%186] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_35 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %cast_36 = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    affine.for %arg40 = 0 to 128 {
      affine.for %arg41 = 0 to 16 {
        %extracted = tensor.extract %189[%arg40, %arg41] : tensor<128x16xi1>
        scf.if %extracted {
          %extracted_37 = tensor.extract %207[%arg40, %arg41] : tensor<128x16xf32>
          %extracted_38 = tensor.extract %80[%arg40, %arg41] : tensor<128x16xi64>
          %208 = arith.index_cast %extracted_38 : i64 to index
          memref.store %extracted_37, %cast_36[%208] : memref<?xf32>
        }
      }
    }
    return
  }
}

