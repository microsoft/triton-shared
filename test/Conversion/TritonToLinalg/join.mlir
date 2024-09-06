// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<128x1xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i32>>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<i32>>
    %7 = tt.join %3, %6 : tensor<128xi32> -> tensor<128x2xi32>
    %8 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = arith.muli %8, %cst : tensor<128x1xi32>
    %10 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>>
    %11 = tt.addptr %10, %9 : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>
    %12 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %14 = tt.broadcast %11 : tensor<128x1x!tt.ptr<i32>> -> tensor<128x2x!tt.ptr<i32>>
    %15 = tt.broadcast %13 : tensor<1x2xi32> -> tensor<128x2xi32>
    %16 = tt.addptr %14, %15 : tensor<128x2x!tt.ptr<i32>>, tensor<128x2xi32>
    tt.store %16, %7 : tensor<128x2x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     [[REINTERPRET_CAST_:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xi32> to memref<128xi32, strided<[1]>>
// CHECK:     [[ALLOC_:%.+]] = memref.alloc() : memref<128xi32>
// CHECK:     memref.copy [[REINTERPRET_CAST_]], [[ALLOC_]] : memref<128xi32, strided<[1]>> to memref<128xi32>
// CHECK:     [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<128xi32>
// CHECK:     [[REINTERPRET_CAST_0_:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1]{{.*}} : memref<*xi32> to memref<128xi32, strided<[1]>>
// CHECK:     [[ALLOC_1_:%.+]] = memref.alloc() : memref<128xi32>
// CHECK:     memref.copy [[REINTERPRET_CAST_0_]], [[ALLOC_1_]] : memref<128xi32, strided<[1]>> to memref<128xi32>
// CHECK:     [[VAR_1_:%.+]] = bufferization.to_tensor [[ALLOC_1_]] restrict writable : memref<128xi32>
// CHECK:     [[VAR_2_:%.+]] = tensor.empty() : tensor<128x2xi32>
// CHECK:     [[INSERTED_SLICE_:%.+]] = tensor.insert_slice [[VAR_0_]] into [[VAR_2_]]{{.*}} : tensor<128xi32> into tensor<128x2xi32>
// CHECK:     [[INSERTED_SLICE_2_:%.+]] = tensor.insert_slice [[VAR_1_]] into [[INSERTED_SLICE_]]{{.*}} : tensor<128xi32> into tensor<128x2xi32>
// CHECK:     [[REINTERPRET_CAST_3_:%.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 2], strides: [%c2, 1]{{.*}} : memref<*xi32> to memref<128x2xi32, strided<[?, 1]>>
// CHECK:     bufferization.materialize_in_destination [[INSERTED_SLICE_2_]] in writable [[REINTERPRET_CAST_3_]] : (tensor<128x2xi32>, memref<128x2xi32, strided<[?, 1]>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
