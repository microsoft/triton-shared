// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func @kernel(%arg0: !tt.ptr<i1>, %arg1: i32, %arg2: i8) {
    %c512 = arith.constant 512 : index
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
    %4 = builtin.unrealized_conversion_cast %3 : !tt.ptr<i8> to memref<*xi8>
    %reinterpret_cast = memref.reinterpret_cast %4 to offset: [%2], sizes: [512], strides: [1] : memref<*xi8> to memref<512xi8, strided<[1], offset: ?>>
    %5 = tensor.empty() : tensor<512xi8>
    %6 = linalg.fill ins(%arg2 : i8) outs(%5 : tensor<512xi8>) -> tensor<512xi8>
    %7 = arith.addi %2, %c512 : index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.minsi %7, %8 : index
    %10 = arith.maxsi %9, %2 : index
    %11 = arith.subi %10, %2 : index
    %extracted_slice = tensor.extract_slice %6[0] [%11] [1] : tensor<512xi8> to tensor<?xi8>
    %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<512xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      [[PARAM_0_:%.+]]: memref<*xi1>,
// CHECK-SAME:      [[PARAM_1_:%.+]]: i32,
// CHECK-SAME:      [[PARAM_2_:%.+]]: i8
// CHECK-SAME:    ) {
// CHECK-DAG:       %[[VAR_0_:.*]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : memref<*xi1> to memref<*xi8>
// CHECK-NOT:       tt.bitcast
