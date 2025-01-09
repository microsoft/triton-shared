#map = affine_map<(d0) -> (d0)>
module {
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
    %1 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %cst = arith.constant 9.900000e+01 : f32
    %dummy_const = arith.constant 1 : i1
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %cst_1 = arith.constant dense<64> : tensor<4xi32>
    %cst_2 = arith.constant dense<3> : tensor<4xi32>
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %2, %arg4 = %2) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_2 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
      %cast = memref.cast %1 : memref<*xf32> to memref<?xf32>
      %8 = bufferization.to_tensor %cast restrict : memref<?xf32>
      %9 = tensor.empty() : tensor<4xf32>
      %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %7 : tensor<4xi32>, tensor<4xi1>) outs(%9 : tensor<4xf32>) {
      ^bb0(%in: i32, %in_4: i1, %out: f32):
        %15 = scf.if %in_4 -> (f32) {
          %16 = arith.index_cast %in : i32 to index
          %extracted = tensor.extract %8[%16] : tensor<?xf32>
          scf.yield %extracted : f32
        } else {
          scf.yield %cst : f32
        }
        linalg.yield %15 : f32
      } -> tensor<4xf32>
      %cast_3 = memref.cast %0 : memref<*xf32> to memref<?xf32>
      %11 = tensor.empty() : tensor<4xi1>
        %alloc = memref.alloc() : memref<4xi1>
      linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%10, %6, %7 : tensor<4xf32>, tensor<4xi32>, tensor<4xi1>) outs(%alloc : memref<4xi1>) {
      ^bb0(%in: f32, %in_4: i32, %in_5: i1, %out: i1):
        %15 = arith.index_cast %in_4 : i32 to index
        %yield = scf.if %in_5 -> i1 {
          memref.store %in, %cast_3[%15] : memref<?xf32>
          scf.yield %dummy_const : i1
        } else {
        scf.yield %dummy_const : i1
        }
        linalg.yield %yield : i1
      }
      %13 = arith.addi %6, %cst_0 : tensor<4xi32>
      %14 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      scf.yield %13, %14 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

