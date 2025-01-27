// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
tt.func public @assert_lol(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
  %1 = tt.splat %0 : i1 -> tensor<1xi1>
  tt.assert %1, "lol": tensor<1xi1>
  tt.return
}

// CHECK: func.func @assert_lol(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK:   cf.assert %0, "Assertion `lol` failed"
// CHECK:   return
// CHECK: }
