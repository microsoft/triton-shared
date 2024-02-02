module {
  module {
    tt.func public @atan2_kernel_0123(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = arith.index_cast %1 : i32 to index
      %5 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%4], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %6 = arith.index_cast %1 : i32 to index
      %7 = arith.addi %6, %c32 : index
      %8 = arith.index_cast %arg3 : i32 to index
      %9 = arith.minsi %7, %8 : index
      %10 = arith.subi %9, %6 : index
      %11 = "tts.load"(%5, %10) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %13 = arith.index_cast %1 : i32 to index
      %14 = arith.addi %13, %c32 : index
      %15 = arith.index_cast %arg3 : i32 to index
      %16 = arith.minsi %14, %15 : index
      %17 = arith.subi %16, %13 : index
      %18 = "tts.load"(%12, %17) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %19 = tt.extern_elementwise %11, %18 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_atan2f"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
      %20 = tts.make_tptr %arg2 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%20, %19) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @pow_kernel_0123(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = arith.index_cast %1 : i32 to index
      %5 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%4], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %6 = arith.index_cast %1 : i32 to index
      %7 = arith.addi %6, %c32 : index
      %8 = arith.index_cast %arg3 : i32 to index
      %9 = arith.minsi %7, %8 : index
      %10 = arith.subi %9, %6 : index
      %11 = "tts.load"(%5, %10) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %13 = arith.index_cast %1 : i32 to index
      %14 = arith.addi %13, %c32 : index
      %15 = arith.index_cast %arg3 : i32 to index
      %16 = arith.minsi %14, %15 : index
      %17 = arith.subi %16, %13 : index
      %18 = "tts.load"(%12, %17) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %19 = tt.extern_elementwise %11, %18 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_powf"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
      %20 = tts.make_tptr %arg2 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%20, %19) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @fabs_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_fabsf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @sin_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_sinf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @cos_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_cosf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @tan_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_tanf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @log_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_logf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @log10_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_log10f"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @log1p_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_log1pf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @exp_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_expf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @exp2_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @erf_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_erff"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @sqrt_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_sqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @rsqrt_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_rsqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @ceil_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_ceilf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @floor_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_floorf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
  module {
    tt.func public @trunc_kernel_012(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: i32) attributes {noinline = false} {
      %c32 = arith.constant 32 : index
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = arith.muli %0, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %1 : i32 to index
      %4 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [%3], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      %5 = arith.index_cast %1 : i32 to index
      %6 = arith.addi %5, %c32 : index
      %7 = arith.index_cast %arg2 : i32 to index
      %8 = arith.minsi %6, %7 : index
      %9 = arith.subi %8, %5 : index
      %10 = "tts.load"(%4, %9) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_dims = array<i64: -9223372036854775808>}> : (tensor<32x!tt.ptr<f32, 1>>, index) -> tensor<32xf32>
      %11 = tt.extern_elementwise %10 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/cuda/lib/libdevice.10.bc", pure = true, symbol = "__nv_truncf"} : (tensor<32xf32>) -> tensor<32xf32>
      %12 = tts.make_tptr %arg1 to sizes: [32], strides: [1], offsets: [%2], shape: [0], order: [] : <f32, 1> to tensor<32x!tt.ptr<f32, 1>>
      "tts.store"(%12, %11) <{static_dims = array<i64>}> : (tensor<32x!tt.ptr<f32, 1>>, tensor<32xf32>) -> ()
      tt.return
    }
  }
}

