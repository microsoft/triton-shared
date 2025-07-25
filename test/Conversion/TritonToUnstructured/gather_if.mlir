// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s
// https://github.com/vllm-project/vllm/blob/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm/attention/ops/triton_decode_attention.py#L472

// -----// IR Dump Before TritonToUnstructured (triton-to-unstructured) //----- //
module {
  tt.func public @_fwd_kernel_stage2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.addptr %arg2, %0 : !tt.ptr<i32>, i32
    %3 = tt.load %2 : !tt.ptr<i32>
    %4 = arith.muli %0, %arg3 : i32
    %5 = arith.muli %1, %arg4 : i32
    %6 = arith.addi %4, %5 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.addi %6, %c32_i32 : i32
    %9 = arith.addi %3, %c3_i32 : i32
    %10 = arith.divsi %9, %c4_i32 : i32
    %11:3 = scf.for %arg8 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg9 = %cst_1, %arg10 = %cst_0, %arg11 = %cst) -> (f32, f32, tensor<32xf32>)  : i32 {
      %20 = arith.muli %10, %arg8 : i32
      %21 = arith.addi %20, %10 : i32
      %22 = arith.minsi %21, %3 : i32
      %23 = arith.cmpi sgt, %22, %20 : i32
      %24:3 = scf.if %23 -> (f32, f32, tensor<32xf32>) {
        %25 = arith.muli %arg8, %arg5 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.addi %7, %26 : index
        %28 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%27], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
        %29 = "tts.load"(%28, %cst_1) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 32>}> : (tensor<32x!tt.ptr<f32>>, f32) -> tensor<32xf32>
        %30 = tt.addptr %arg0, %8 : !tt.ptr<f32>, i32
        %31 = tt.addptr %30, %25 : !tt.ptr<f32>, i32
        %32 = tt.load %31 : !tt.ptr<f32>
        %33 = arith.maxnumf %32, %arg10 : f32
        %34 = arith.subf %arg10, %33 : f32
        %35 = math.exp %34 : f32
        %36 = tt.splat %35 : f32 -> tensor<32xf32>
        %37 = arith.mulf %arg11, %36 : tensor<32xf32>
        %38 = arith.subf %32, %33 : f32
        %39 = math.exp %38 : f32
        %40 = tt.splat %39 : f32 -> tensor<32xf32>
        %41 = arith.mulf %40, %29 : tensor<32xf32>
        %42 = arith.addf %37, %41 : tensor<32xf32>
        %43 = arith.mulf %arg9, %35 : f32
        %44 = arith.addf %43, %39 : f32
        scf.yield %44, %33, %42 : f32, f32, tensor<32xf32>
      } else {
        scf.yield %arg9, %arg10, %arg11 : f32, f32, tensor<32xf32>
      }
      scf.yield %24#0, %24#1, %24#2 : f32, f32, tensor<32xf32>
    }
    %12 = arith.muli %0, %arg6 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.muli %1, %arg7 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.addi %13, %15 : index
    %17 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%16], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
    %18 = tt.splat %11#0 : f32 -> tensor<32xf32>
    %19 = arith.divf %11#2, %18 : tensor<32xf32>
    "tts.store"(%17, %19) <{static_mask_dims = array<i64: 32>}> : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>) -> ()
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store