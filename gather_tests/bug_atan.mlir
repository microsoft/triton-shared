module {
  tt.func public @atan2_kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %12 = tt.load %11, %6 : tensor<32x!tt.ptr<f32>>
    %13 = tt.extern_elementwise %9, %12 {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %15, %13 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}