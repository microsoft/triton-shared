// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = "tts.make_unstructured_tptr"(%arg0, %4) : (!tt.ptr<f32>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.load %7, %6 : tensor<1024x!tt.ptr<f32>>
    %9 = "tts.make_unstructured_tptr"(%arg1, %4) : (!tt.ptr<f32>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.load %9, %6 : tensor<1024x!tt.ptr<f32>>
    %11 = arith.addf %8, %10 : tensor<1024xf32>
    %12 = "tts.make_unstructured_tptr"(%arg2, %4) : (!tt.ptr<f32>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    tt.store %12, %11, %6 : tensor<1024x!tt.ptr<f32>>
    return
  }
}

// CHECK:   func.func public @add_kernel_01234(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32)
// CHECK-DAG: builtin.unrealized_conversion_cast %arg2 : memref<*xf32> to !tt.ptr<f32>
// CHECK-DAG: builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !tt.ptr<f32>
// CHECK-DAG: builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !tt.ptr<f32>
