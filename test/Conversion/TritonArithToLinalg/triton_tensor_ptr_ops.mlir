// Test all triton ops on tensor of pointers are converted to linalg.generic
// Original triton program:
//    @triton.jit
//    def tensor_ptr(in_ptr0, out_ptr0):
//        ints = tl.load(in_ptr0 + tl.arange(0, 16)).to(tl.int64)
//        ptrs = ints.to(tl.pointer_type(tl.int32))
//        vals = tl.load(ptrs)
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        ints_2 = out_ptrs.to(tl.int64) + vals
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        out_ptrs_i64 = out_ptrs.to(tl.pointer_type(tl.int64))
//        out_ptrs_i64 += 2
//        out_ptrs_i32 = out_ptrs_i64.to(tl.pointer_type(tl.int32))
//        tl.store(out_ptrs_i32, ints_2.to(tl.int32))

// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg=true" %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>>
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %9 = tt.ptr_to_int %8 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %10 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64>
    %11 = arith.addi %9, %10 : tensor<16xi64>
    %12 = tt.bitcast %8 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %13 = tt.addptr %12, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %14 = tt.bitcast %13 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %15 = arith.trunci %11 : tensor<16xi64> to tensor<16xi32>
    tt.store %14, %15 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: tt.addptr %in, %in_0 : !tt.ptr<i32>, i32
// CHECK: tt.load %in : !tt.ptr<i32>
// CHECK: tt.int_to_ptr %in : i64 -> !tt.ptr<i32>
// CHECK: tt.load %in : !tt.ptr<i32>
// CHECK: tt.addptr %in, %in_0 : !tt.ptr<i32>, i32
// CHECK: tt.ptr_to_int %in : !tt.ptr<i32> -> i64
// CHECK: tt.bitcast %in : !tt.ptr<i32> -> !tt.ptr<i64>
// CHECK: tt.addptr %in, %in_0 : !tt.ptr<i64>, i32
// CHECK: tt.bitcast %in : !tt.ptr<i64> -> !tt.ptr<i32>
// CHECK: tt.store %in, %in_0 : !tt.ptr<i32>
