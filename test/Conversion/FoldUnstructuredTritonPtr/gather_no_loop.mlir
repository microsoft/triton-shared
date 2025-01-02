// RUN: triton-shared-opt --fold-unstructured-triton-ptr %s | FileCheck %s

module {
  tt.func public @gather_simple_no_loop(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<64xi32>
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = arith.divsi %0, %cst_0 : tensor<64xi32>
    %2 = arith.addi %1, %cst : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.load %4 : tensor<64x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %7, %5 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-COUNT-2: "tts.make_unstructured_tptr"(%arg{{[0-9]+}}
// CHECK-NOT:    "tts.make_unstructured_tptr"(%arg{{[0-9]+}}