%12:3 = "scf.for"(%6, %5, %4, %2, %10, %8) ({
^bb0(%arg11: index, %arg12: tensor<128x128xf32>, %arg13: index, %arg14: index):
  %18 = "arith.addi"(<<UNKNOWN SSA VALUE>>, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  %19 = "tts.make_tptr"(<<UNKNOWN SSA VALUE>>, %18) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, order = array<i32>, sizes = array<i64: 128, 128>, static_offsets = array<i64: -9223372036854775808, 0>, static_shape = array<i64: 0, 0>, static_strides = array<i64: 1, 1>}> : (!tt.ptr<f32, 1>, index) -> tensor<128x128x!tt.ptr<f32, 1>>
  %20 = "tts.load"(%19) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32, 1>>) -> tensor<128x128xf32>
  %21 = "linalg.generic"(%20, %20) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg15: f32, %arg16: f32):
    %26 = "math.exp"(%arg15) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
    "linalg.yield"(%26) : (f32) -> ()
  }) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  %22 = "linalg.generic"(<<UNKNOWN SSA VALUE>>, %21, <<UNKNOWN SSA VALUE>>) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):
    %26 = "arith.addf"(%arg15, %arg16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%26) : (f32) -> ()
  }) : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  %23 = "arith.index_cast"(<<UNKNOWN SSA VALUE>>) : (index) -> i32
  %24 = "tt.addptr"(<<UNKNOWN SSA VALUE>>, %23) : (!tt.ptr<f32, 1>, i32) -> !tt.ptr<f32, 1>
  %25 = "arith.addi"(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "scf.yield"(%22, %24, %25) : (tensor<128x128xf32>, !tt.ptr<f32, 1>, index) -> ()
}) : (index, index, index, tensor<128x128xf32>, index, index) -> (tensor<128x128xf32>, index, index)
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = arith.muli %arg8, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %5 = builtin.unrealized_conversion_cast %arg1, %4 : memref<*xf32>, index to index
    %6:3 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %5, %arg14 = %3) -> (tensor<128x128xf32>, index, index) {
      %10 = arith.addi %arg14, %c128 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf32>
      memref.copy %reinterpret_cast_0, %alloc : memref<128x128xf32, strided<[1, 1], offset: ?>> to memref<128x128xf32>
      %11 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
      %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<128x128xf32>) outs(%11 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = math.exp %in : f32
        linalg.yield %16 : f32
      } -> tensor<128x128xf32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %12 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg12 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %16 = arith.addf %in, %in_1 : f32
        linalg.yield %16 : f32
      } -> tensor<128x128xf32>
      %14 = builtin.unrealized_conversion_cast %arg13, %arg11 : index, index to index
      %15 = arith.addi %arg14, %arg11 : index
      scf.yield %13, %14, %15 : tensor<128x128xf32>, index, index
    }
    %7 = arith.muli %arg8, %arg3 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.addi %8, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %6#0 in writable %reinterpret_cast : (tensor<128x128xf32>, memref<128x128xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

