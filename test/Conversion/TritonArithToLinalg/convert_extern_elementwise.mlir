// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @atan2_kernel_0123(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : (i32) -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg3 : (i32) -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xf32>
    %10 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %11 = tt.addptr %10, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xf32>
    %13 = tt.extern_elementwise %9, %12 {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %14 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    tt.store %15, %13 {cache = 1 : i32, evict = 1 : i32} : tensor<32xf32>
    tt.return
  }
}
// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @atan2_kernel_0123
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]], [[VAR_2:%.+]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_3:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_4:%.+]] = math.atan2 [[in_1]], [[in_2]] : f32
// CHECK:          linalg.yield [[VAR_4]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @sin_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : (i32) -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : (i32) -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xf32>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_sinf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    tt.store %12, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<32xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @sin_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.sin [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>
