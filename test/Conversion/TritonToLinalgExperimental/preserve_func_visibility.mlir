// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0 : !tt.ptr<i32>
    %2 = arith.cmpi eq, %0, %c0_i32 : i32
    %3 = scf.if %2 -> (i32) {
      %4 = tt.call @test_core.add_fn_return__i32_i32__(%1, %0) : (i32, i32) -> i32
      scf.yield %4 : i32
    } else {
      scf.yield %1 : i32
    }
    tt.store %arg0, %3 : !tt.ptr<i32>
    tt.return
  }
  tt.func private @test_core.add_fn_return__i32_i32__(%arg0: i32, %arg1: i32) -> i32 attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg1, %c0_i32 : i32
    cf.cond_br %0, ^bb1(%c1_i32 : i32), ^bb1(%c2_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb0
    %2 = arith.addi %arg0, %1 : i32
    tt.return %2 : i32
  }
}

// Public is implicit
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32)

// CHECK-LABEL:  func.func private @test_core.add_fn_return__i32_i32__
// CHECK-SAME:   ([[PARAM_0_:%.+]]: i32, [[PARAM_1_:%.+]]: i32) -> i32
