#map = affine_map<(d0) -> (d0)>
module {
  func.func @gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.fill ins(%c4_i32 : i32) outs(%0 : tensor<4xi32>) -> tensor<4xi32>
    %c64_i32 = arith.constant 64 : i32
    %2 = tensor.empty() : tensor<4xi32>
    %3 = linalg.fill ins(%c64_i32 : i32) outs(%2 : tensor<4xi32>) -> tensor<4xi32>
    %c3_i32 = arith.constant 3 : i32
    %4 = tensor.empty() : tensor<4xi32>
    %5 = linalg.fill ins(%c3_i32 : i32) outs(%4 : tensor<4xi32>) -> tensor<4xi32>
    %c1_i32 = arith.constant 1 : i32
    %6 = tensor.empty() : tensor<4xi32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %9 = linalg.index 0 : index
      %10 = arith.index_cast %9 : index to i32
      linalg.yield %10 : i32
    } -> tensor<4xi32>
    %8:3 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %7, %arg10 = %7, %arg11 = %c0) -> (tensor<4xi32>, tensor<4xi32>, index)  : i32 {
      %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %5 : tensor<4xi32>, tensor<4xi32>) outs(%arg9 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.divsi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %10 = tensor.empty() : tensor<4xi32>
      %11 = linalg.fill ins(%arg8 : i32) outs(%10 : tensor<4xi32>) -> tensor<4xi32>
      %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %11 : tensor<4xi32>, tensor<4xi32>) outs(%9 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %13 = tensor.empty() : tensor<4xi1>
      %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %3 : tensor<4xi32>, tensor<4xi32>) outs(%13 : tensor<4xi1>) {
      ^bb0(%in: i32, %in_0: i32, %out: i1):
        %41 = arith.cmpi slt, %in, %in_0 : i32
        linalg.yield %41 : i1
      } -> tensor<4xi1>
      %15 = tensor.empty() : tensor<4xi64>
      %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<4xi32>) outs(%15 : tensor<4xi64>) {
      ^bb0(%in: i32, %out: i64):
        %41 = arith.extsi %in : i32 to i64
        linalg.yield %41 : i64
      } -> tensor<4xi64>
      %17 = "tts.make_unstructured_tptr"(%arg0, %16) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      %18 = tt.load %17, %14 : tensor<4x!tt.ptr<f32>>
      %19 = tensor.empty() : tensor<4xi64>
      %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<4xi32>) outs(%19 : tensor<4xi64>) {
      ^bb0(%in: i32, %out: i64):
        %41 = arith.extsi %in : i32 to i64
        linalg.yield %41 : i64
      } -> tensor<4xi64>
      %21 = "tts.make_unstructured_tptr"(%arg1, %20) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      tt.store %21, %18 : tensor<4x!tt.ptr<f32>>
      %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %1 : tensor<4xi32>, tensor<4xi32>) outs(%12 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg10, %1 : tensor<4xi32>, tensor<4xi32>) outs(%arg10 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %24 = arith.addi %arg8, %c1_i32 : i32
      %25 = arith.addi %arg11, %c4 : index
      %26:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %22, %arg14 = %23, %arg15 = %25) -> (tensor<4xi32>, tensor<4xi32>, index)  : i32 {
        %41 = arith.addi %arg12, %c1_i32 : i32
        %42 = arith.muli %24, %41 : i32
        %43 = tensor.empty() : tensor<4xi32>
        %44 = linalg.fill ins(%42 : i32) outs(%43 : tensor<4xi32>) -> tensor<4xi32>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %44 : tensor<4xi32>, tensor<4xi32>) outs(%arg13 : tensor<4xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
          %59 = arith.divsi %in, %in_0 : i32
          linalg.yield %59 : i32
        } -> tensor<4xi32>
        %46 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%45, %11 : tensor<4xi32>, tensor<4xi32>) outs(%45 : tensor<4xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
          %59 = arith.addi %in, %in_0 : i32
          linalg.yield %59 : i32
        } -> tensor<4xi32>
        %47 = tensor.empty() : tensor<4xi1>
        %48 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%46, %3 : tensor<4xi32>, tensor<4xi32>) outs(%47 : tensor<4xi1>) {
        ^bb0(%in: i32, %in_0: i32, %out: i1):
          %59 = arith.cmpi slt, %in, %in_0 : i32
          linalg.yield %59 : i1
        } -> tensor<4xi1>
        %49 = tensor.empty() : tensor<4xi64>
        %50 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%46 : tensor<4xi32>) outs(%49 : tensor<4xi64>) {
        ^bb0(%in: i32, %out: i64):
          %59 = arith.extsi %in : i32 to i64
          linalg.yield %59 : i64
        } -> tensor<4xi64>
        %51 = "tts.make_unstructured_tptr"(%arg0, %50) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
        %52 = tt.load %51, %48 : tensor<4x!tt.ptr<f32>>
        %53 = tensor.empty() : tensor<4xi64>
        %54 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%46 : tensor<4xi32>) outs(%53 : tensor<4xi64>) {
        ^bb0(%in: i32, %out: i64):
          %59 = arith.extsi %in : i32 to i64
          linalg.yield %59 : i64
        } -> tensor<4xi64>
        %55 = "tts.make_unstructured_tptr"(%arg1, %54) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
        tt.store %55, %52 : tensor<4x!tt.ptr<f32>>
        %56 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%46, %1 : tensor<4xi32>, tensor<4xi32>) outs(%46 : tensor<4xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
          %59 = arith.addi %in, %in_0 : i32
          linalg.yield %59 : i32
        } -> tensor<4xi32>
        %57 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg14, %1 : tensor<4xi32>, tensor<4xi32>) outs(%arg14 : tensor<4xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
          %59 = arith.addi %in, %in_0 : i32
          linalg.yield %59 : i32
        } -> tensor<4xi32>
        %58 = arith.addi %arg15, %c4 : index
        scf.yield %56, %57, %58 : tensor<4xi32>, tensor<4xi32>, index
      }
      %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%26#0, %5 : tensor<4xi32>, tensor<4xi32>) outs(%26#0 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.divsi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%27, %11 : tensor<4xi32>, tensor<4xi32>) outs(%27 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %29 = tensor.empty() : tensor<4xi1>
      %30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%28, %3 : tensor<4xi32>, tensor<4xi32>) outs(%29 : tensor<4xi1>) {
      ^bb0(%in: i32, %in_0: i32, %out: i1):
        %41 = arith.cmpi slt, %in, %in_0 : i32
        linalg.yield %41 : i1
      } -> tensor<4xi1>
      %31 = tensor.empty() : tensor<4xi64>
      %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%28 : tensor<4xi32>) outs(%31 : tensor<4xi64>) {
      ^bb0(%in: i32, %out: i64):
        %41 = arith.extsi %in : i32 to i64
        linalg.yield %41 : i64
      } -> tensor<4xi64>
      %33 = "tts.make_unstructured_tptr"(%arg0, %32) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      %34 = tt.load %33, %30 : tensor<4x!tt.ptr<f32>>
      %35 = tensor.empty() : tensor<4xi64>
      %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%28 : tensor<4xi32>) outs(%35 : tensor<4xi64>) {
      ^bb0(%in: i32, %out: i64):
        %41 = arith.extsi %in : i32 to i64
        linalg.yield %41 : i64
      } -> tensor<4xi64>
      %37 = "tts.make_unstructured_tptr"(%arg1, %36) : (!tt.ptr<f32>, tensor<4xi64>) -> tensor<4x!tt.ptr<f32>>
      tt.store %37, %34 : tensor<4x!tt.ptr<f32>>
      %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%28, %1 : tensor<4xi32>, tensor<4xi32>) outs(%28 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%26#1, %1 : tensor<4xi32>, tensor<4xi32>) outs(%26#1 : tensor<4xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %41 = arith.addi %in, %in_0 : i32
        linalg.yield %41 : i32
      } -> tensor<4xi32>
      %40 = arith.addi %26#2, %c4 : index
      scf.yield %38, %39, %40 : tensor<4xi32>, tensor<4xi32>, index
    }
    return
  }
}