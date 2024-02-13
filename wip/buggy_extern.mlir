module {
  tt.func public @fabs_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
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
    %10 = tt.extern_elementwise %9 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_fabsf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32, 1>>, tensor<32xi32>
    tt.store %12, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<32xf32>
    tt.return
  }
}
