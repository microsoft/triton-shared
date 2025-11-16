// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
  %5 = tt.make_range {end = 1152 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 1
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,1]
  %7 = tt.broadcast %6 : tensor<1x128xi32> -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,6]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [1,6]
  // mixed use
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = arith.sitofp %14 : tensor<256x128xi32> to tensor<256x128xbf16>
  tt.store %18, %20 : tensor<256x128x!tt.ptr<bf16>>
  tt.return
  }
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: memref<*xbf16>, %[[ARG1:.*]]: memref<*xbf16>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1024 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 512 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 6 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<256x128xi32>) -> tensor<256x128xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [256, 1] : tensor<256xi32> into tensor<256x1xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<256x1xi32>) outs(%[[EMPTY_2]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_1]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_3]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[INDEX_CAST_1]], %[[CONSTANT_0]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_2]] {{\[\[}}0, 1]] output_shape [1, 128] : tensor<128xi32> into tensor<1x128xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x128xi32>) outs(%[[EMPTY_4]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_3]], %[[FILL_0]] : tensor<256x128xi32>, tensor<256x128xi32>) outs(%[[GENERIC_3]] : tensor<256x128xi32>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]], %[[GENERIC_4]] : tensor<256x128xi32>, tensor<256x128xi32>) outs(%[[GENERIC_1]] : tensor<256x128xi32>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             linalg.yield %[[ADDI_2]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [6656], sizes: [256, 128], strides: [1, 6] : memref<*xbf16> to memref<256x128xbf16, strided<[1, 6], offset: 6656>>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<256x128xbf16, strided<[1, 6], offset: 6656>> to memref<256x128xbf16, strided<[1, ?], offset: 6656>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<256x128xbf16, strided<[1, 6], offset: 6656>> to memref<256x128xbf16>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<256x128xbf16> to tensor<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[CAST_0]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 6656>>) -> ()
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<256x128xbf16>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_5]] : tensor<256x128xi32>) outs(%[[EMPTY_5]] : tensor<256x128xbf16>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: bf16):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_12]] : i32 to bf16
// CHECK:             linalg.yield %[[SITOFP_0]] : bf16
// CHECK:           } -> tensor<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination %[[GENERIC_6]] in writable %[[CAST_0]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, ?], offset: 6656>>) -> ()
// CHECK:           return
// CHECK:         }
